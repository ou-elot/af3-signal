import numpy as np
import pandas as pd
import pytest

from af3_signal.analysis import aggregate, analyze
from af3_signal.constructs import prepare_inputs
from af3_signal.io import cached_download, read_table, table, write_json
from af3_signal.references import process_references
from af3_signal.reporting import report
from test_inputs_import import import_fixture_rows


def test_network_independent_pending_path(synthetic_run, monkeypatch):
    config, root = synthetic_run
    def no_network(*args, **kwargs):
        raise AssertionError('Offline analysis attempted network access')
    monkeypatch.setattr('urllib.request.urlopen', no_network)
    process_references(config)
    prepare_inputs(config)
    scores = analyze(config)
    assert scores.empty
    comparisons = read_table(root / 'results/comparisons.csv')
    assert set(comparisons.status) == {'PENDING'}
    assert not read_table(root / 'data/processed/predictions_normalized.csv').shape[0]
    assert set(read_table(root / 'results/pending_jobs.csv').status) == {'PENDING'}
    # Synthetic reference is explicitly labelled in its title and temporary directory.
    path = report(config)
    assert 'BLOCKED / PENDING' in path.read_text(encoding='utf-8')
    assert len(list((root / 'figures').glob('*.svg'))) == 8


def test_full_import_analysis_and_incomplete_mask_sensitivity(synthetic_run, fixture_cif):
    config, root = synthetic_run
    process_references(config)
    prepare_inputs(config)
    import_fixture_rows(config, root, fixture_cif)
    scores = analyze(config)
    native = scores[(scores.variant == 'native') & (scores.flank_id == 'unflanked')].iloc[0]
    assert native.primary_error == pytest.approx(0, abs=1e-7)
    assert native.low_variable_count == native.eligible_count == 10
    assert native.native_low_paired_error == pytest.approx(0, abs=1e-7)
    assert native.ca_rmsd < 1e-7
    assert native.target_low_fraction_known == 1
    assert set(read_table(root / 'results/pending_jobs.csv').status) == {'IMPORTED'}
    research_report = report(config).read_text(encoding='utf-8')
    assert 'SYNTHETIC SOFTWARE TEST ONLY' in research_report
    assert 'DESCRIPTIVE_COMPLETE_IMPORTED_SEEDS' in research_report
    constructs = read_table(root / 'inputs/constructs.csv')
    missing_id = constructs[(constructs.variant == 'shuffle') & (constructs.flank_id == 'unflanked')].iloc[0].construct_id
    import_fixture_rows(config, root, fixture_cif, missing_construct=missing_id)
    scores = analyze(config)
    incomplete = scores[scores.construct_id == missing_id].iloc[0]
    assert np.isnan(incomplete.primary_error)
    assert incomplete.coverage_status == 'BELOW_THRESHOLD'
    assert np.isfinite(incomplete.common_mask_error)
    unflanked = scores[scores.flank_id == 'unflanked']
    assert unflanked.common_mask_count.nunique() == 1
    assert set(unflanked.common_mask_count) == {7}
    residues = pd.read_csv(root / 'data/processed/prediction_mapping.csv')
    missing = residues[(residues.construct_id == missing_id) & (residues.construct_position == 4)]
    assert len(missing) == 1 and missing.label_seq.isna().all()
    errors = pd.read_csv(root / 'results/per_residue.csv')
    assert {'phi_pred', 'psi_pred', 'phi_ref', 'psi_ref', 'error', 'plddt'} <= set(errors.columns)
    assert errors.phi_pred.notna().any() and errors.phi_ref.notna().any()


def test_missing_confidence_stays_missing_and_report_rejects_stale_manifest(synthetic_run, fixture_cif):
    config, root = synthetic_run
    process_references(config)
    prepare_inputs(config)
    rows = import_fixture_rows(config, root, fixture_cif)
    for row in rows:
        row['confidence_source'] = 'missing'
        row['confidence_evidence'] = ''
    table(root / 'predictions/manifest.csv', rows)
    scores = analyze(config)
    assert scores.target_confidence_count.eq(0).all()
    assert scores.low_variable_error.isna().all()
    assert scores.native_low_paired_error.isna().all()
    assert scores.primary_error.notna().all()
    normalized = read_table(root / 'data/processed/predictions_normalized.csv')
    assert set(normalized.validation_status) == {'MISSING_CONFIDENCE'}
    rows[0]['inference_source'] = 'changed after analysis'
    table(root / 'predictions/manifest.csv', rows)
    with pytest.raises(ValueError, match='changed since analysis'):
        report(config)


def test_unresolved_reference_position_has_explicit_missingness(synthetic_run, fixture_cif):
    from af3_signal.io import sha256
    config, root = synthetic_run
    reference = root / 'data/raw/TEST.cif'
    fixture_cif(reference, config['proteins']['tau']['segment'], missing=(1,), author_start=306)
    write_json(reference.with_suffix('.cif.provenance.json'), {
        'url': 'https://files.rcsb.org/download/TEST.cif',
        'retrieved_utc': 'SYNTHETIC SOFTWARE TEST ONLY', 'sha256': sha256(reference)})
    process_references(config)
    prepare_inputs(config)
    import_fixture_rows(config, root, fixture_cif)
    analyze(config)
    errors = pd.read_csv(root / 'results/per_residue.csv')
    missing = errors[errors.canonical_position == 306]
    assert missing.phi_ref.isna().all() and missing.psi_ref.isna().all()
    assert set(missing.phi_missing_reason_ref) == {'reference_residue_unresolved'}
    assert not missing.primary_eligible.any()


def test_seed_weighting_and_construct_not_residue_n(synthetic_run):
    config, root = synthetic_run
    rows = []
    for variant, cid in [('native', 'c1'), ('shuffle', 'c2')]:
        for seed, values in [(11, [0, 0, 0]), (22, [100])]:
            for index, value in enumerate(values):
                rows.append({'protein': 'tau', 'reference': 'TEST', 'flank_id': 'unflanked',
                             'construct_id': cid, 'variant': variant, 'seed': seed, 'sample_index': index,
                             **{metric: value for metric in ['primary_error', 'common_mask_error', 'low_variable_error', 'native_low_paired_error']}})
    aggregate(config, pd.DataFrame(rows), None)
    agg = pd.read_csv(root / 'results/construct_scores.csv')
    assert set(agg.mean_error) == {50}  # not 25 from treating samples/seeds interchangeably
    comparisons = pd.read_csv(root / 'results/comparisons.csv')
    row = comparisons[(comparisons.flank_id == 'unflanked') & (comparisons.metric == 'primary_error')].iloc[0]
    assert row.independent_shuffle_constructs == 1
    assert row.status == 'PARTIAL_OR_INSUFFICIENT'  # undeclared extra seed isn't a complete planned comparison


def test_cache_integrity_and_missing_offline(tmp_path):
    with pytest.raises(ValueError, match='Offline cache missing'):
        cached_download('https://example.invalid/test', tmp_path / 'missing.cif', offline=True)
    path = tmp_path / 'tracked.cif'
    path.write_text('tampered', encoding='utf-8')
    write_json(path.with_suffix('.cif.provenance.json'), {'url': 'https://example.invalid/test', 'sha256': 'wrong'})
    with pytest.raises(ValueError, match='integrity/source mismatch'):
        cached_download('https://example.invalid/test', path, offline=True)
