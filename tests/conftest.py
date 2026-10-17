"""Transparent synthetic coordinates for SOFTWARE TESTS ONLY, never research outputs."""
import json
from pathlib import Path

import numpy as np
import pytest
from Bio.SeqUtils import seq3

from af3_signal.io import initialize, load_config, sha256, write_json


def write_fixture_cif(path, sequence, missing=(), broken_after=None, label_offset=0,
                      author_start=101, chain='Z', omit_atom=None, altloc=False):
    """Arbitrary connected curve, NOT an AF3 inference or experimental structure."""
    header = ['data_SYNTHETIC_SOFTWARE_TEST_ONLY', '#', '_struct.title "SYNTHETIC SOFTWARE TEST ONLY"',
              'loop_', '_struct_asym.id', '_struct_asym.entity_id', f'{chain} 1', '#',
              'loop_', '_entity_poly_seq.entity_id', '_entity_poly_seq.num', '_entity_poly_seq.mon_id']
    header += [f'1 {i + 1 + label_offset} {seq3(aa).upper()}' for i, aa in enumerate(sequence)]
    header += ['#', 'loop_']
    fields = ['group_PDB', 'id', 'type_symbol', 'label_atom_id', 'label_alt_id', 'label_comp_id',
              'label_asym_id', 'label_entity_id', 'label_seq_id', 'pdbx_PDB_ins_code', 'Cartn_x',
              'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv', 'auth_seq_id', 'auth_comp_id',
              'auth_asym_id', 'auth_atom_id', 'pdbx_PDB_model_num']
    header += ['_atom_site.' + f for f in fields]
    lines = []
    count = 0
    for i, aa in enumerate(sequence, 1):
        if i in missing:
            continue
        origin = np.array([i * 2., np.sin(i), np.cos(i)])
        if broken_after is not None and i > broken_after:
            origin += [20, 0, 0]
        for atom, xyz in [('N', origin), ('CA', origin + [.6, .7, .2]), ('C', origin + [1.2, .1, .1])]:
            if (i, atom) == omit_atom:
                continue
            alternatives = [('A', .5, xyz), ('B', .5, xyz + [0, .3, 0])] if altloc and i == 2 else [('.', 1., xyz)]
            for alt, occ, coords in alternatives:
                count += 1
                lines.append(f'ATOM {count} {atom[0]} {atom} {alt} {seq3(aa).upper()} {chain} 1 {i + label_offset} ? '
                             f'{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {occ} 40 '
                             f'{i + author_start - 1} {seq3(aa).upper()} AUTHOR {atom} 1')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(header + lines + ['#', '']), encoding='utf-8')
    return path


@pytest.fixture
def fixture_cif():
    return write_fixture_cif


@pytest.fixture
def synthetic_run(tmp_path):
    config = load_config('configs/smoke.yaml')
    config['run_dir'] = str(tmp_path / 'SYNTHETIC_SOFTWARE_TEST_ONLY')
    config['software_test_only'] = True
    config['proteins']['tau']['segment'] = 'ACDEFGHIKLMN'
    config['proteins']['tau']['canonical_end'] = 317
    config['proteins']['tau']['references'] = [{'id': 'TEST', 'role': 'primary', 'doi': '10.test/fixture'}]
    config['analysis']['min_residues'] = 3
    config['flanks'] = [{'id': 'unflanked', 'left': '', 'right': ''}, {'id': 'test_flanks', 'left': 'GS', 'right': 'PE'}]
    root = initialize(config)
    raw = root / 'data/raw'
    cif = write_fixture_cif(raw / 'TEST.cif', config['proteins']['tau']['segment'], author_start=306)
    write_json(cif.with_suffix('.cif.provenance.json'), {'url': 'https://files.rcsb.org/download/TEST.cif',
               'retrieved_utc': 'SOFTWARE TEST ONLY', 'sha256': sha256(cif)})
    metadata = raw / 'TEST.metadata.json'
    write_json(metadata, {'rcsb_primary_citation': {'pdbx_database_id_DOI': '10.test/fixture',
               'title': 'SYNTHETIC SOFTWARE TEST ONLY', 'year': 2026},
               'rcsb_accession_info': {'initial_release_date': '2026-01-01'},
               'struct': {'title': 'SYNTHETIC SOFTWARE TEST ONLY'}})
    write_json(metadata.with_suffix('.json.provenance.json'), {'url': 'https://data.rcsb.org/rest/v1/core/entry/TEST',
               'retrieved_utc': 'SOFTWARE TEST ONLY', 'sha256': sha256(metadata)})
    return config, root
