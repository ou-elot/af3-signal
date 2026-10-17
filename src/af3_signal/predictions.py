"""Strict manual AF3 adapter; no inference, website automation, or B-factor guessing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constructs import MANIFEST_COLUMNS
from .io import initialize, read_table, sha256, table
from .structure import angles, map_prediction, parse_cif


def confidence_values(data, residues, confidence_path=None, source='missing', evidence=''):
    """Mean N/CA/C pLDDT only when all three atoms have documented confidence."""
    n = len(data['_atom_site.id'])
    metric_names = [x.lower() for x in data.get('_ma_qa_metric.name', [])]
    af3_software = any('alphafold' in x.lower() and '3' in x for x in data.get('_software.name', []))
    metadata_plddt = af3_software and any('plddt' in x for x in metric_names)
    documented_b = metadata_plddt or (source == 'af3_bfactor' and bool(evidence.strip()))
    b = np.array([float(x) if x not in ('?', '.') else np.nan
                  for x in data.get('_atom_site.B_iso_or_equiv', ['?'] * n)])
    values = np.full(n, np.nan)
    convention = 'missing'
    if confidence_path:
        obj = json.loads(Path(confidence_path).read_text())
        if 'atom_plddts' not in obj or 'atom_chain_ids' not in obj:
            raise ValueError('Confidence JSON needs atom_plddts and atom_chain_ids; summary-only JSON is not per-residue confidence')
        raw, chains = obj['atom_plddts'], obj['atom_chain_ids']
        if len(raw) != n or len(chains) != n:
            raise ValueError(f'Misaligned atom confidence: JSON length != mmCIF atom count {n}')
        labels = data['_atom_site.label_asym_id']
        authors = data.get('_atom_site.auth_asym_id', labels)
        if list(chains) != labels and list(chains) != authors:
            raise ValueError('Misaligned confidence atom_chain_ids versus mmCIF atom order')
        values = np.array([np.nan if x is None else float(x) for x in raw])
        convention = 'af3_json_atom_order_N_CA_C_mean'
        # AF3 JSON has no per-atom names; agreeing with documented AF3 B values
        # supplies a stronger check than count/chain agreement alone.
        if documented_b:
            valid = np.isfinite(values) & np.isfinite(b)
            if valid.any() and not np.allclose(values[valid], b[valid], atol=0.11, rtol=0):
                raise ValueError('JSON atom pLDDT does not match documented mmCIF pLDDT; wrong sample/order?')
    elif documented_b:
        values = b
        convention = 'documented_af3_bfactor_N_CA_C_mean'
    elif source != 'missing':
        raise ValueError('Absent/undocumented confidence: supply full AF3 JSON or af3_bfactor with confidence_evidence')
    finite = values[np.isfinite(values)]
    if ((finite < 0) | (finite > 100)).any() or np.isinf(values).any():
        raise ValueError('pLDDT values must be 0..100 or missing')
    result = {}
    for r in residues:
        indices = [r.atom_rows.get(a) for a in ('N', 'CA', 'C')]
        complete = all(i is not None for i in indices)
        atom_values = values[indices] if complete else np.array([np.nan])
        result[r.position] = float(atom_values.mean()) if np.isfinite(atom_values).all() else np.nan
    return result, convention


def resolve_source(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def normalize_predictions(config, manifest=None):
    root = initialize(config)
    manifest = Path(manifest) if manifest else root / 'predictions/manifest.csv'
    constructs = read_table(root / 'inputs/constructs.csv').set_index('construct_id')
    source = read_table(manifest)
    missing = set(MANIFEST_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f'Prediction manifest missing columns: {sorted(missing)}')
    normalized, mapping, residue_rows, objects = [], [], [], {}
    sample_keys, sample_ids, source_hashes, model_settings = set(), set(), set(), set()
    for _, row in source.iterrows():
        sid, cid = str(row.sample_id), str(row.construct_id)
        try:
            if not sid or sid in sample_ids:
                raise ValueError('sample_id must be nonempty and unique')
            if cid not in constructs.index:
                raise ValueError(f'Unknown construct_id {cid}')
            c = constructs.loc[cid]
            seed, index = int(row.seed), int(row.sample_index)
            if str(seed) != str(row.seed) or str(index) != str(row.sample_index) or index < 0:
                raise ValueError('Seed/sample_index must be integers; sample_index >= 0')
            if seed not in config['prediction_seeds']:
                raise ValueError(f'Seed {seed} was not declared in configuration')
            if (cid, seed, index) in sample_keys:
                raise ValueError('Duplicate construct/seed/sample index')
            if row.model_name != 'alphafold3':
                raise ValueError('This adapter is AF3-only; other model families require separate analyses')
            for field in ['model_version', 'checkpoint', 'msa_settings', 'template_settings', 'inference_date', 'inference_source']:
                if not str(row[field]).strip():
                    raise ValueError(f'Missing inference provenance: {field}; use explicit unknown where unavailable')
            for actual, planned in [('msa_settings', 'msa'), ('template_settings', 'templates'),
                                    ('model_version', 'version'), ('checkpoint', 'checkpoint')]:
                expected = str(config['model'][planned])
                observed = str(row[actual])
                if 'unknown' not in expected.lower() and 'unknown' not in observed.lower() and observed != expected:
                    raise ValueError(f'Inference setting {actual}={observed!r} differs from frozen plan {expected!r}; use a separate preregistered run')
            if row.input_sha256 != c.input_sha256:
                raise ValueError('Input SHA-256 mismatch: sample not traced to exact generated AF3 input JSON')
            actual_input = root / c.input_path
            if sha256(actual_input) != c.input_sha256:
                raise ValueError('Generated input JSON changed since preparation')
            path = resolve_source(root, str(row.cif_path))
            file_hash = sha256(path)
            if file_hash in source_hashes:
                raise ValueError('Duplicate coordinate file content; do not count top-ranked copies as new samples')
            residues, data = parse_cif(path)
            chain = str(row.chain_id)
            if len({r.label_chain for r in residues}) != 1:
                raise ValueError('Expected a monomer output with one polymer chain; import a documented monomer prediction')
            offset = int(row.label_offset or 0)
            selected = map_prediction(residues, data, c.sequence, chain, offset)
            conf_path = resolve_source(root, str(row.confidence_json)) if row.confidence_json else None
            confidence, convention = confidence_values(data, selected, conf_path,
                                                       str(row.confidence_source), str(row.confidence_evidence))
            by_position = {r.position: r for r in selected}
            aa = {a['position']: a for a in angles(selected, config['analysis']['peptide_bond_max_angstrom'])}
            spec = config['proteins'][c.protein]
            target_positions = set(range(int(c.flank_offset) + 1, int(c.flank_offset) + int(c.target_length) + 1))
            for pos, expected in enumerate(c.sequence, 1):
                r = by_position.get(pos)
                segment_pos = pos - int(c.flank_offset) if pos in target_positions else None
                canonical = spec['canonical_start'] + segment_pos - 1 if segment_pos else None
                info = {'sample_id': sid, 'construct_id': cid, 'protein': c.protein, 'construct_position': pos,
                        'segment_position': segment_pos, 'canonical_position': canonical, 'is_target': pos in target_positions,
                        'expected_amino_acid': expected, 'mapping_basis': c.mapping_basis,
                        **(r.identifiers() if r else {k: '' for k in ['label_chain', 'auth_chain', 'label_seq', 'auth_seq', 'insertion_code', 'amino_acid', 'altloc']})}
                mapping.append(info)
                angle = aa.get(pos, {'phi': np.nan, 'psi': np.nan, 'phi_missing_reason': 'missing_residue', 'psi_missing_reason': 'missing_residue'})
                residue_rows.append({**info, **{k: angle[k] for k in ['phi', 'psi', 'phi_missing_reason', 'psi_missing_reason']},
                                     'plddt': confidence.get(pos, np.nan), 'confidence_convention': convention})
            missing_conf = sum(not np.isfinite(confidence.get(pos, np.nan)) for pos in target_positions)
            normalized.append({**row.to_dict(), 'coordinate_sha256': file_hash,
                               'confidence_sha256': sha256(conf_path) if conf_path else '',
                               'confidence_convention': convention, 'target_missing_confidence': missing_conf,
                               'validation_status': 'MISSING_CONFIDENCE' if missing_conf else 'OK',
                               'provenance_unknown_fields': ';'.join(k for k in MANIFEST_COLUMNS if 'unknown' in str(row[k]).lower()),
                               'json_alignment_check': 'atom_count_and_chain_order; B-factor crosscheck when documented'})
            model_settings.add(tuple(str(row[k]) for k in ['model_version', 'checkpoint', 'msa_settings', 'template_settings']))
            sample_ids.add(sid)
            sample_keys.add((cid, seed, index))
            source_hashes.add(file_hash)
            objects[sid] = selected
        except (ValueError, OSError, KeyError) as exc:
            raise ValueError(f'Prediction {sid or "<unnamed>"}: {exc}') from exc
    if len(model_settings) > 1:
        raise ValueError('Mixed model/checkpoint/MSA/template settings in one run; use separate run directories')
    norm = table(root / 'data/processed/predictions_normalized.csv', normalized,
                 MANIFEST_COLUMNS + ['coordinate_sha256', 'confidence_sha256', 'confidence_convention', 'target_missing_confidence',
                                     'validation_status', 'provenance_unknown_fields', 'json_alignment_check'])
    table(root / 'data/processed/prediction_mapping.csv', mapping,
          ['sample_id', 'construct_id', 'protein', 'construct_position', 'segment_position', 'canonical_position', 'is_target',
           'expected_amino_acid', 'mapping_basis', 'label_chain', 'auth_chain', 'label_seq', 'auth_seq', 'insertion_code', 'amino_acid', 'altloc'])
    frame = table(root / 'data/processed/prediction_angles.csv', residue_rows,
                  ['sample_id', 'construct_id', 'protein', 'construct_position', 'segment_position', 'canonical_position', 'is_target',
                   'expected_amino_acid', 'mapping_basis', 'label_chain', 'auth_chain', 'label_seq', 'auth_seq', 'insertion_code',
                   'amino_acid', 'altloc', 'phi', 'psi', 'phi_missing_reason', 'psi_missing_reason', 'plddt', 'confidence_convention'])
    jobs = read_table(root / 'inputs/jobs.csv')
    for i, job in jobs.iterrows():
        count = sum(1 for _, s in norm.iterrows() if s.construct_id == job.construct_id and int(s.seed) == int(job.seed))
        jobs.loc[i, 'actual_samples'] = count
        jobs.loc[i, 'status'] = 'IMPORTED' if count else 'PENDING'
    table(root / 'results/pending_jobs.csv', jobs)
    return norm, frame, objects
