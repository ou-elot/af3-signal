"""Predetermined-mask comparisons and construct-level descriptive effects."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest, initialize, read_table, sha256, table, write_json
from .metrics import angular_error, beta_fraction, contact_overlap, contacts, mask_score, rmsd
from .predictions import normalize_predictions
from .references import process_references, reference_objects


SCORE_METRICS = ['primary_error', 'common_mask_error', 'low_variable_error', 'native_low_paired_error']
SCORE_COLUMNS = ['sample_id', 'construct_id', 'protein', 'reference', 'flank_id', 'variant', 'seed', 'sample_index',
                 'primary_error', 'primary_status', 'eligible_count', 'valid_count', 'coverage', 'coverage_status',
                 'common_mask_error', 'common_mask_count', 'low_variable_error', 'low_variable_count',
                 'native_low_paired_error', 'native_low_paired_count', 'native_low_pair_status',
                 'target_confidence_count', 'target_confidence_missing', 'target_low_fraction_known', 'target_mean_plddt',
                 'flank_mean_plddt', 'flank_confidence_count', 'beta_fraction', 'ca_rmsd', 'ca_count', 'ca_status',
                 'reference_contacts', 'contact_recall', 'contact_jaccard', 'contact_status']


def analyze(config, manifest=None):
    root = initialize(config)
    refs = process_references(config)
    norm, pred, objects = normalize_predictions(config, manifest)
    audit_path = root / 'data/processed/exposure_audit.csv'
    audit = pd.read_csv(audit_path)
    for column in ['model_version', 'checkpoint', 'msa_settings', 'template_settings', 'inference_source']:
        audit['actual_' + column] = ';'.join(sorted(set(norm[column].astype(str)))) if len(norm) else 'PENDING'
    if len(norm):
        setting = str(norm.template_settings.iloc[0])
        audit['actual_template_exposure'] = ('No inference templates declared; training exposure remains unknown'
                                             if setting == 'disabled' else 'UNKNOWN; inspect actual inference provenance')
    table(audit_path, audit)
    constructs = read_table(root / 'inputs/constructs.csv').set_index('construct_id')
    ref_objects = reference_objects(config)
    a = config['analysis']
    min_n = a['min_residues']
    threshold = a['low_confidence_threshold']
    masks, eligible = [], {}
    for protein in config['enabled_proteins']:
        spec = config['proteins'][protein]
        for reference in spec['references']:
            rid = reference['id']
            rr = refs[(refs.reference == rid) & refs.representative].copy()
            good = rr[np.isfinite(rr.phi) & np.isfinite(rr.psi)]
            # Segment terminals excluded for every context, even when flanks supply neighbors.
            positions = set(good.position.astype(int)) & set(range(spec['canonical_start'] + 1,
                            spec['canonical_start'] + len(spec['segment']) - 1))
            eligible[rid] = sorted(positions)
            for pos in range(spec['canonical_start'], spec['canonical_start'] + len(spec['segment'])):
                masks.append({'protein': protein, 'reference': rid, 'canonical_position': pos,
                              'primary_eligible': pos in positions,
                              'reason': '' if pos in positions else 'reference_angles_missing_or_segment_terminal'})
    table(root / 'results/predetermined_masks.csv', masks)
    per_residue, sample_data = [], []
    for _, sample in norm.iterrows():
        c = constructs.loc[sample.construct_id]
        spec = config['proteins'][c.protein]
        target = pred[(pred.sample_id == sample.sample_id) & pred.is_target].copy()
        for reference in spec['references']:
            rid = reference['id']
            rr = refs[(refs.reference == rid) & refs.representative]
            # Include all target residues in export; only declared positions enter the primary score.
            combined = target.merge(rr[['position', 'phi', 'psi', 'phi_missing_reason', 'psi_missing_reason']],
                                    left_on='canonical_position', right_on='position', how='left', suffixes=('_pred', '_ref'))
            for angle in ['phi', 'psi']:
                combined[angle + '_missing_reason_ref'] = combined[angle + '_missing_reason_ref'].fillna('reference_residue_unresolved')
            combined['error'] = angular_error(combined.phi_pred, combined.psi_pred, combined.phi_ref, combined.psi_ref)
            combined['reference'] = rid
            combined['primary_eligible'] = combined.canonical_position.isin(eligible[rid])
            combined['low_confidence'] = combined.plddt < threshold
            per_residue.extend(combined.to_dict('records'))
            indexed = combined.set_index('canonical_position').reindex(eligible[rid])
            finite = np.isfinite(indexed.error.to_numpy())
            valid = set(indexed.index[finite].astype(int))
            sample_data.append({'sample': sample, 'construct': c, 'reference': rid,
                                'data': indexed, 'target': target, 'valid': valid})
    common = {}
    for entry in sample_data:
        c = entry['construct']
        key = (c.protein, entry['reference'], c.flank_id)
        common[key] = common[key] & entry['valid'] if key in common else set(entry['valid'])
    native_positions = {}
    for entry in sample_data:
        s, c = entry['sample'], entry['construct']
        if c.variant == 'native':
            frame = entry['data']
            key = (c.protein, entry['reference'], c.flank_id, int(s.seed), int(s.sample_index))
            native_positions[key] = set(frame.index[frame.plddt < threshold].astype(int))
    scores = []
    for entry in sample_data:
        s, c, rid = entry['sample'], entry['construct'], entry['reference']
        frame, target = entry['data'], entry['target']
        mask_common = frame.index.isin(common[(c.protein, rid, c.flank_id)])
        low_mask = (frame.plddt < threshold).to_numpy()
        key = (c.protein, rid, c.flank_id, int(s.seed), int(s.sample_index))
        paired = native_positions.get(key)
        paired_mask = frame.index.isin(paired) if paired is not None else np.zeros(len(frame), dtype=bool)
        n_valid, n_eligible = len(entry['valid']), len(frame)
        coverage = n_valid / n_eligible if n_eligible else np.nan
        primary = mask_score(frame.error, np.ones(len(frame), dtype=bool), min_n)
        flank = pred[(pred.sample_id == s.sample_id) & ~pred.is_target]
        known = target.plddt.notna()
        # Longer-range metrics use fixed reference target C-alpha positions. No variable dropping.
        reference_ca = {r.position: r.atoms['CA'] for r in ref_objects[rid] if 'CA' in r.atoms and not r.ambiguous}
        predicted_ca = {r.position - int(c.flank_offset) + config['proteins'][c.protein]['canonical_start'] - 1: r.atoms['CA']
                        for r in objects[s.sample_id] if 'CA' in r.atoms and not r.ambiguous}
        ca_complete = set(reference_ca) <= set(predicted_ca)
        ca_positions = sorted(reference_ca)
        ref_contacts = contacts(reference_ca, a['contact_distance_angstrom'], a['contact_min_sequence_separation'])
        if ca_complete:
            predicted_contacts = contacts({k: predicted_ca[k] for k in ca_positions}, a['contact_distance_angstrom'], a['contact_min_sequence_separation'])
            overlap = contact_overlap(predicted_contacts, ref_contacts)
            ca_rmsd = rmsd([predicted_ca[k] for k in ca_positions], [reference_ca[k] for k in ca_positions])
        else:
            overlap = {'contact_recall': np.nan, 'contact_jaccard': np.nan, 'contact_status': 'missing_predicted_ca_fixed_mask'}
            ca_rmsd = np.nan
        scores.append({'sample_id': s.sample_id, 'construct_id': s.construct_id, 'protein': c.protein, 'reference': rid,
                       'flank_id': c.flank_id, 'variant': c.variant, 'seed': int(s.seed), 'sample_index': int(s.sample_index),
                       'primary_error': primary,
                       'primary_status': 'OK' if np.isfinite(primary) else 'insufficient_or_incomplete_fixed_mask',
                       'eligible_count': n_eligible, 'valid_count': n_valid, 'coverage': coverage,
                       'coverage_status': 'OK' if np.isfinite(coverage) and coverage >= a['min_coverage'] else 'BELOW_THRESHOLD',
                       'common_mask_error': mask_score(frame.error, mask_common, min_n), 'common_mask_count': int(mask_common.sum()),
                       'low_variable_error': mask_score(frame.error, low_mask, min_n), 'low_variable_count': int(low_mask.sum()),
                       'native_low_paired_error': mask_score(frame.error, paired_mask, min_n),
                       'native_low_paired_count': int(paired_mask.sum()),
                       'native_low_pair_status': 'native_sample_missing' if paired is None else ('OK' if paired_mask.sum() >= min_n and np.isfinite(frame.error.to_numpy()[paired_mask]).all() else 'insufficient_or_incomplete_mask'),
                       'target_confidence_count': int(known.sum()), 'target_confidence_missing': int((~known).sum()),
                       'target_low_fraction_known': float((target.loc[known, 'plddt'] < threshold).mean()) if known.any() else np.nan,
                       'target_mean_plddt': target.plddt.mean(), 'flank_mean_plddt': flank.plddt.mean(),
                       'flank_confidence_count': int(flank.plddt.notna().sum()),
                       'beta_fraction': beta_fraction(frame.phi_pred, frame.psi_pred, a['beta_phi'], a['beta_psi']),
                       'ca_rmsd': ca_rmsd, 'ca_count': len(ca_positions), 'ca_status': 'OK' if ca_complete else 'incomplete_fixed_mask',
                       'reference_contacts': len(ref_contacts), **overlap})
    score_df = table(root / 'results/sample_scores.csv', scores, SCORE_COLUMNS)
    residue_columns = [k for k in pred.columns if k not in ['phi', 'psi', 'phi_missing_reason', 'psi_missing_reason']]
    table(root / 'results/per_residue.csv', per_residue,
          residue_columns + ['position', 'phi_pred', 'psi_pred', 'phi_ref', 'psi_ref',
                               'phi_missing_reason_pred', 'psi_missing_reason_pred', 'phi_missing_reason_ref', 'psi_missing_reason_ref',
                               'error', 'reference', 'primary_eligible', 'low_confidence'])
    aggregate(config, score_df, constructs)
    manifest_path = Path(manifest) if manifest else root / 'predictions/manifest.csv'
    artifacts = [p for folder in ['data/processed', 'results', 'inputs']
                 for p in (root / folder).rglob('*') if p.is_file() and p.name != 'analysis_state.json']
    write_json(root / 'results/analysis_state.json', {
        'config_sha256': digest(config),
        'manifest_path_relative_to_run': Path(os.path.relpath(manifest_path, root)).as_posix(),
        'manifest_sha256': sha256(manifest_path),
        'analysis_artifact_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in sorted(artifacts)},
    })
    return score_df


def aggregate(config, scores, constructs):
    root = initialize(config)
    aggregates, comparisons = [], []
    keys = ['protein', 'reference', 'flank_id', 'construct_id', 'variant']
    if len(scores):
        for group, frame in scores.groupby(keys, sort=True):
            for metric in SCORE_METRICS:
                # Every imported sample must have a score; missing samples are not silently discarded.
                per_seed = frame.groupby('seed')[metric].agg(lambda s: s.mean() if s.notna().all() else np.nan)
                all_imported_valid = per_seed.notna().all()
                value = float(per_seed.mean()) if all_imported_valid else np.nan
                aggregates.append({**dict(zip(keys, group)), 'metric': metric, 'mean_error': value,
                                   'imported_samples': len(frame), 'imported_seeds': len(per_seed),
                                   'valid_samples': int(frame[metric].notna().sum()),
                                   'all_planned_seeds_imported': set(frame.seed) == set(config['prediction_seeds']),
                                   'status': 'OK' if all_imported_valid else 'insufficient_or_incomplete_sample_scores'})
    agg = table(root / 'results/construct_scores.csv', aggregates,
                keys + ['metric', 'mean_error', 'imported_samples', 'imported_seeds', 'valid_samples', 'all_planned_seeds_imported', 'status'])
    for protein in config['enabled_proteins']:
        for ref in config['proteins'][protein]['references']:
            for flank in config['flanks']:
                for metric in SCORE_METRICS:
                    subset = agg[(agg.protein == protein) & (agg.reference == ref['id']) & (agg.flank_id == flank['id']) & (agg.metric == metric)]
                    native = subset[subset.variant == 'native'].mean_error
                    controls = subset[subset.variant == 'shuffle'].mean_error
                    complete = (len(native) == 1 and native.notna().all() and len(controls) == config['n_shuffles']
                                and controls.notna().all() and subset.all_planned_seeds_imported.all())
                    observed_native = native.iloc[0] if len(native) else np.nan
                    descriptive_difference = observed_native - controls.mean() if len(controls) and controls.notna().any() else np.nan
                    comparisons.append({'protein': protein, 'reference': ref['id'], 'flank_id': flank['id'], 'metric': metric,
                                        'native_error': observed_native, 'shuffle_mean': controls.mean(), 'shuffle_sd': controls.std(ddof=1),
                                        'native_minus_shuffle': descriptive_difference,
                                        'independent_shuffle_constructs': int(controls.notna().sum()),
                                        'planned_shuffle_constructs': config['n_shuffles'],
                                        'status': 'DESCRIPTIVE_COMPLETE_IMPORTED_SEEDS' if complete else ('PENDING' if not len(subset) else 'PARTIAL_OR_INSUFFICIENT'),
                                        'effect_direction': 'negative favors native; descriptive sequence-order-null effect'})
    table(root / 'results/comparisons.csv', comparisons)
