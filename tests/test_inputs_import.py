from collections import Counter
import json

import numpy as np
import pandas as pd
import pytest

from af3_signal.constructs import MANIFEST_COLUMNS, prepare_inputs, shuffles
from af3_signal.io import initialize, load_config, read_table, sha256, table, write_json
from af3_signal.predictions import confidence_values, normalize_predictions
from af3_signal.references import process_references
from af3_signal.structure import map_prediction, parse_cif


def test_config_canonical_tau_and_shuffles():
    config = load_config('configs/pilot.yaml')
    tau = config['proteins']['tau']
    assert tau['canonical_start'] == 306 and tau['canonical_end'] == 378
    assert len(tau['segment']) == 73
    seqs = shuffles(tau['segment'], 5, 123)
    assert seqs == shuffles(tau['segment'], 5, 123)
    assert len(set(seqs + [tau['segment']])) == 6
    assert all(Counter(s) == Counter(tau['segment']) for s in seqs)
    assert all(len(f['left']) == len(f['right']) == 20 for f in config['flanks'][1:])
    with pytest.raises(ValueError, match='enough unique'):
        shuffles('AAAA', 2, 1)


def test_construct_offsets_context_matching_and_freeze(synthetic_run):
    config, root = synthetic_run
    process_references(config)
    assert prepare_inputs(config) == (4, 4)
    before = sha256(root / 'inputs/constructs.csv')
    prepare_inputs(config)
    assert sha256(root / 'inputs/constructs.csv') == before
    constructs = read_table(root / 'inputs/constructs.csv')
    assert constructs.groupby('shuffle_index').segment_sequence.nunique().eq(1).all()
    mapping = pd.read_csv(root / 'inputs/construct_reference_mapping.csv')
    flank_ids = set(constructs[constructs.flank_id == 'test_flanks'].construct_id)
    flanked = mapping[mapping.construct_id.isin(flank_ids)]
    assert (flanked.construct_position == flanked.segment_position + 2).all()
    assert set(constructs[constructs.variant == 'shuffle'].mapping_basis) == {'segment_position_sequence_order_null'}
    config['analysis']['low_confidence_threshold'] = 49
    with pytest.raises(ValueError, match='Configuration changed'):
        initialize(config)


def test_confidence_is_never_guessed_and_json_alignment(tmp_path, fixture_cif):
    path = fixture_cif(tmp_path / 'fixture.cif', 'ACDE')
    residues, data = parse_cif(path)
    selected = map_prediction(residues, data, 'ACDE', 'Z')
    values, convention = confidence_values(data, selected)
    assert convention == 'missing' and all(np.isnan(x) for x in values.values())
    with pytest.raises(ValueError, match='Absent/undocumented'):
        confidence_values(data, selected, source='af3_bfactor')
    values, _ = confidence_values(data, selected, source='af3_bfactor', evidence='SYNTHETIC SOFTWARE TEST ONLY')
    assert values[1] == 40
    j = tmp_path / 'confidence.json'
    obj = {'atom_plddts': [40] * 12, 'atom_chain_ids': ['Z'] * 12}
    write_json(j, obj)
    values, _ = confidence_values(data, selected, j)
    assert values[3] == 40
    obj['atom_chain_ids'][0] = 'WRONG'
    write_json(j, obj)
    with pytest.raises(ValueError, match='Misaligned'):
        confidence_values(data, selected, j)
    obj['atom_chain_ids'] = ['Z'] * 11
    write_json(j, obj)
    with pytest.raises(ValueError, match='length'):
        confidence_values(data, selected, j)
    obj = {'atom_plddts': [90] * 12, 'atom_chain_ids': ['Z'] * 12}
    write_json(j, obj)
    with pytest.raises(ValueError, match='does not match'):
        confidence_values(data, selected, j, source='af3_bfactor', evidence='SYNTHETIC SOFTWARE TEST ONLY')
    write_json(j, {'ptm': .5})
    with pytest.raises(ValueError, match='summary-only'):
        confidence_values(data, selected, j)


def import_fixture_rows(config, root, fixture_cif, missing_construct=None):
    constructs = read_table(root / 'inputs/constructs.csv')
    rows = []
    for _, c in constructs.iterrows():
        path = root / f'predictions/{c.construct_id}_SOFTWARE_TEST_ONLY.cif'
        fixture_cif(path, c.sequence, missing=(4,) if c.construct_id == missing_construct else ())
        rows.append({'sample_id': c.construct_id + '_TEST_ONLY', 'construct_id': c.construct_id,
                     'seed': 11, 'sample_index': 0, 'cif_path': path.relative_to(root).as_posix(),
                     'confidence_json': '', 'chain_id': 'Z', 'label_offset': 0,
                     'model_name': 'alphafold3', 'model_version': 'SYNTHETIC_TEST_ONLY', 'checkpoint': 'SYNTHETIC_TEST_ONLY',
                     'input_sha256': c.input_sha256, 'msa_settings': 'none', 'template_settings': 'disabled',
                     'inference_date': 'SOFTWARE TEST ONLY', 'inference_source': 'SYNTHETIC FIXTURE - NO INFERENCE',
                     'confidence_source': 'af3_bfactor', 'confidence_evidence': 'SYNTHETIC SOFTWARE TEST ONLY'})
    table(root / 'predictions/manifest.csv', rows, MANIFEST_COLUMNS)
    return rows


def test_import_rejects_wrong_hash_and_duplicate_samples(synthetic_run, fixture_cif):
    config, root = synthetic_run
    process_references(config)
    prepare_inputs(config)
    rows = import_fixture_rows(config, root, fixture_cif)
    norm, residues, _ = normalize_predictions(config)
    assert len(norm) == 4
    assert residues.plddt.eq(40).all()
    rows[0]['input_sha256'] = 'bad'
    table(root / 'predictions/manifest.csv', rows, MANIFEST_COLUMNS)
    with pytest.raises(ValueError, match='Input SHA-256 mismatch'):
        normalize_predictions(config)
    rows[0]['input_sha256'] = read_table(root / 'inputs/constructs.csv').iloc[0].input_sha256
    rows.append(dict(rows[0]))
    table(root / 'predictions/manifest.csv', rows, MANIFEST_COLUMNS)
    with pytest.raises(ValueError, match='unique'):
        normalize_predictions(config)
    rows = rows[:-1]
    rows[0]['template_settings'] = 'automatic'
    table(root / 'predictions/manifest.csv', rows, MANIFEST_COLUMNS)
    with pytest.raises(ValueError, match='differs from frozen plan'):
        normalize_predictions(config)
