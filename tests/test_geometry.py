import copy

import numpy as np
import pytest

from af3_signal.metrics import angular_error, beta_fraction, circular_delta, contact_overlap, contacts, mask_score, rmsd
from af3_signal.structure import angles, dihedral, map_prediction, parse_cif, reference_mapping


def test_wraparound_missing_and_identical():
    assert circular_delta(-179, 179) == 2
    assert np.isnan(circular_delta(np.nan, 0))
    assert angular_error(-100, 150, -100, 150) == 0
    assert np.isnan(mask_score([np.nan], [True], 1))
    assert np.isnan(mask_score([], [], 1))
    assert np.isnan(beta_fraction([], []))


def test_dihedral_matches_known_right_angle():
    assert dihedral([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1]]) == pytest.approx(-90)


def test_flank_numbering_missing_residues_do_not_shift(tmp_path, fixture_cif):
    seq = 'GSACDEPE'
    path = fixture_cif(tmp_path / 'fixture.cif', seq, missing=(4,), label_offset=10, author_start=501)
    residues, data = parse_cif(path)
    selected = map_prediction(residues, data, seq, 'Z', label_offset=10)
    assert [(r.position, r.label_seq, r.auth_seq) for r in selected][:3] == [(1, 11, '501'), (2, 12, '502'), (3, 13, '503')]
    assert selected[3].position == 5
    angle = {a['position']: a for a in angles(selected)}
    assert np.isnan(angle[3]['psi'])
    assert np.isnan(angle[5]['phi'])
    assert angle[3]['psi_missing_reason'] == 'terminal_or_missing_neighbor'
    with pytest.raises(ValueError, match='mismatch|outside'):
        map_prediction(residues, data, seq, 'Z', label_offset=0)


def test_chain_break_and_missing_atom(tmp_path, fixture_cif):
    path = fixture_cif(tmp_path / 'break.cif', 'ACDEFG', broken_after=3, omit_atom=(2, 'CA'))
    residues, data = parse_cif(path)
    selected = map_prediction(residues, data, 'ACDEFG', 'Z')
    result = {a['position']: a for a in angles(selected)}
    assert result[3]['psi_missing_reason'] == 'peptide_chain_break'
    assert result[4]['phi_missing_reason'] == 'peptide_chain_break'
    assert result[2]['phi_missing_reason'] == 'missing_backbone_atom'
    assert np.isnan(result[1]['phi']) and np.isnan(result[6]['psi'])


def test_altloc_and_insertion_ambiguity(tmp_path, fixture_cif):
    path = fixture_cif(tmp_path / 'alt.cif', 'ACDE', altloc=True)
    residues, data = parse_cif(path)
    selected = map_prediction(residues, data, 'ACDE', 'Z')
    assert selected[1].altloc == 'A'  # tied occupancy resolved lexicographically
    assert not selected[1].ambiguous
    selected[1].ambiguous = True
    result = angles(selected)
    assert result[1]['phi_missing_reason'] == 'insertion_or_identity_ambiguity'
    assert result[0]['psi_missing_reason'] == 'insertion_or_identity_ambiguity'


def test_canonical_mapping_and_identical_structure(tmp_path, fixture_cif):
    seq = 'ACDEFGHIKLMN'
    path = fixture_cif(tmp_path / 'reference.cif', seq, author_start=900)
    residues, data = parse_cif(path)
    selected = reference_mapping(residues, data, seq, 306)
    assert selected[0].position == 306 and selected[-1].position == 317
    assert selected[0].auth_seq == '900'  # numbering is not guessed
    aa = angles(selected)
    values = [a for a in aa if np.isfinite(a['phi']) and np.isfinite(a['psi'])]
    assert len(values) == 10
    errors = angular_error([a['phi'] for a in values], [a['psi'] for a in values],
                           [a['phi'] for a in values], [a['psi'] for a in values])
    assert np.max(errors) < 1e-9
    residues[3].aa = 'W'
    with pytest.raises(ValueError, match='sequence mismatch'):
        reference_mapping(residues, data, seq, 306)


def test_contacts_empty_and_proper_rotation():
    assert np.isnan(contact_overlap(set(), set())['contact_recall'])
    coords = {1: np.array([0, 0, 0]), 2: np.array([0, 1, 0]), 5: np.array([0, 0, 2])}
    assert contacts(coords) == {(1, 5)}  # > 3, not >= 3
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert rmsd(reference @ rotation + 10, reference) < 1e-10
    assert np.isnan(rmsd([], []))
