"""Reproducible exact-composition controls and identity-neutral model inputs."""
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd

from .io import digest, initialize, sha256, table, write_json


MANIFEST_COLUMNS = ['sample_id', 'construct_id', 'seed', 'sample_index', 'cif_path',
                    'confidence_json', 'chain_id', 'label_offset', 'model_name', 'model_version',
                    'checkpoint', 'input_sha256', 'msa_settings', 'template_settings',
                    'inference_date', 'inference_source', 'confidence_source', 'confidence_evidence']


def shuffles(sequence, count, seed):
    rng = np.random.default_rng(seed)
    found, seen = [], {sequence}
    for _ in range(max(1000, count * 1000)):
        candidate = ''.join(rng.permutation(list(sequence)))
        if candidate not in seen:
            assert len(candidate) == len(sequence) and Counter(candidate) == Counter(sequence)
            found.append(candidate)
            seen.add(candidate)
        if len(found) == count:
            return found
    raise ValueError('Could not generate enough unique non-native shuffles')


def prepare_inputs(config):
    root = initialize(config)
    # Reconcile only managed, previous inputs. Never remove user prediction files.
    previous = root / 'inputs/constructs.csv'
    previous_ids = set(pd.read_csv(previous).construct_id) if previous.exists() else set()
    constructs, jobs, mappings = [], [], []
    refs = pd.read_csv(root / 'data/processed/reference_mapping.csv', keep_default_na=False)
    refs = refs[refs.representative.astype(str).str.lower() == 'true']
    fasta = []
    for protein in config['enabled_proteins']:
        spec = config['proteins'][protein]
        # Stable per-protein stream: enabling A-beta does not change tau shuffles.
        rng_seed = (config['shuffle_seed'] + int(hashlib.sha256(protein.encode()).hexdigest()[:8], 16)) % (2**32)
        variants = [spec['segment']] + shuffles(spec['segment'], config['n_shuffles'], rng_seed)
        for index, segment in enumerate(variants):
            for flank in config['flanks']:
                sequence = flank['left'] + segment + flank['right']
                if set(sequence) - set('ACDEFGHIKLMNPQRSTVWY'):
                    raise ValueError(f'Nonstandard sequence for {protein}/{flank["id"]}')
                identity = {'protein': protein, 'index': index, 'flank': flank['id'], 'sequence': sequence}
                cid = 'c' + digest(identity)[:12]
                input_obj = {'name': cid, 'sequences': [{'protein': {'id': 'A', 'sequence': sequence,
                             'unpairedMsa': '', 'pairedMsa': '', 'templates': []}}],
                             'modelSeeds': config['prediction_seeds'], 'dialect': 'alphafold3', 'version': 1}
                if config['model']['msa'] != 'none' or config['model']['templates'] != 'disabled':
                    raise ValueError('This pilot emitter supports explicit MSA-free/template-free inputs only')
                input_path = root / f'inputs/af3/{cid}.json'
                write_json(input_path, input_obj)
                record = {'construct_id': cid, 'protein': protein, 'variant': 'native' if index == 0 else 'shuffle',
                          'shuffle_index': index, 'flank_id': flank['id'], 'sequence': sequence,
                          'segment_sequence': segment, 'left_flank': flank['left'], 'right_flank': flank['right'],
                          'flank_offset': len(flank['left']), 'target_length': len(segment),
                          'flank_disorder': 'assumed_candidate_not_checked', 'shuffle_rng_seed': rng_seed,
                          'global_shuffle_seed': config['shuffle_seed'], 'input_sha256': sha256(input_path),
                          'input_path': input_path.relative_to(root).as_posix(),
                          'mapping_basis': 'native_identity' if index == 0 else 'segment_position_sequence_order_null'}
                constructs.append(record)
                fasta.append(f'>{cid}\n{sequence}\n')
                (root / f'inputs/{cid}.fasta').write_text(f'>{cid}\n{sequence}\n', encoding='utf-8')
                for seed in config['prediction_seeds']:
                    jobs.append({'construct_id': cid, 'protein': protein, 'seed': seed,
                                 'input_sha256': record['input_sha256'], 'status': 'PENDING',
                                 'expected_samples': config['expected_samples_per_seed'], 'actual_samples': 0})
                for pos in range(1, len(segment) + 1):
                    canonical = spec['canonical_start'] + pos - 1
                    matching = refs[(refs.protein == protein) & (refs.canonical_position == canonical)]
                    for ref in spec['references']:
                        hits = matching[matching.reference == ref['id']]
                        hit = hits.iloc[0].to_dict() if len(hits) else {}
                        mappings.append({'construct_id': cid, 'protein': protein, 'reference': ref['id'],
                                         'canonical_position': canonical, 'segment_position': pos,
                                         'construct_position': pos + record['flank_offset'],
                                         'construct_amino_acid': segment[pos - 1], 'native_amino_acid': spec['segment'][pos - 1],
                                         'mapping_basis': record['mapping_basis'],
                                         **{'reference_' + k: hit.get(k, '') for k in
                                            ['label_chain', 'auth_chain', 'label_seq', 'auth_seq', 'insertion_code']}})
    table(root / 'inputs/constructs.csv', constructs)
    table(root / 'inputs/jobs.csv', jobs)
    table(root / 'inputs/construct_reference_mapping.csv', mappings)
    (root / 'inputs/constructs.fasta').write_text(''.join(fasta), encoding='utf-8')
    current_ids = {c['construct_id'] for c in constructs}
    for old in previous_ids - current_ids:
        for path in [root / f'inputs/{old}.fasta', root / f'inputs/af3/{old}.json']:
            if path.exists():
                path.unlink()
    manifest = root / 'predictions/manifest.csv'
    if not manifest.exists():
        table(manifest, [], MANIFEST_COLUMNS)
    # Fingerprint the exact design and thresholds for reports/imports.
    write_json(root / 'inputs/design.json', {'config_sha256': digest(config), 'construct_count': len(constructs),
               'construct_seed_jobs': len(jobs), 'planned_samples_only': len(jobs) * config['expected_samples_per_seed'],
               'null': 'Exact-composition segment permutations mapped by segment position; conditional sequence-order null',
               'inference': 'Descriptive effects only; no formal significance test',
               'primary_mask': 'Reference both-angle positions intersected with internal segment positions before importing predictions',
               'missing_predictions': 'Primary score withheld unless every predetermined position is available; common-mask sensitivity separate'})
    return len(constructs), len(jobs)
