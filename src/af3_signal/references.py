"""RCSB retrieval and sequence-verified single-chain reference tables."""
import json

import numpy as np
import pandas as pd
from Bio import SeqIO

from .io import cached_download, initialize, sha256, table
from .structure import angles, parse_cif, reference_mapping


DOCS = {
    'af3_input.md': 'https://raw.githubusercontent.com/google-deepmind/alphafold3/main/docs/input.md',
    'af3_output.md': 'https://raw.githubusercontent.com/google-deepmind/alphafold3/main/docs/output.md',
    'af3_readme.md': 'https://raw.githubusercontent.com/google-deepmind/alphafold3/main/README.md',
    'af3_paper.html': 'https://www.nature.com/articles/s41586-024-07487-w',
    'af3_citation.json': 'https://api.crossref.org/works/10.1038/s41586-024-07487-w',
}


def fetch_references(config, offline=False):
    root = initialize(config)
    raw = root / 'data/raw'
    sources = []
    for filename, url in DOCS.items():
        meta = cached_download(url, raw / 'sources' / filename, offline)
        sources.append({'document': filename, 'url': url, 'sha256': meta['sha256'], 'retrieved_utc': meta['retrieved_utc']})
    citation = json.loads((raw / 'sources/af3_citation.json').read_text())['message']
    if citation['DOI'].lower() != '10.1038/s41586-024-07487-w' or 'AlphaFold 3' not in citation['title'][0]:
        raise ValueError('AF3 source citation identity mismatch')
    table(root / 'data/processed/source_provenance.csv', sources)
    for protein in config['enabled_proteins']:
        spec = config['proteins'][protein]
        fasta = raw / f'{spec["canonical_accession"]}.fasta'
        cached_download(spec['canonical_fasta_url'], fasta, offline)
        seq = str(SeqIO.read(fasta, 'fasta').seq)
        start = spec.get('precursor_start', spec['canonical_start']) - 1
        if len(seq) != spec['canonical_length'] or seq[start:start + len(spec['segment'])] != spec['segment']:
            raise ValueError(f'Canonical FASTA mismatch for {protein}; expected {spec["canonical_accession"]} and configured segment')
        for ref in spec['references']:
            rid = ref['id']
            cached_download(f'https://files.rcsb.org/download/{rid}.cif', raw / f'{rid}.cif', offline)
            cached_download(f'https://data.rcsb.org/rest/v1/core/entry/{rid}', raw / f'{rid}.metadata.json', offline)
    return process_references(config)


def process_references(config):
    root = initialize(config)
    raw = root / 'data/raw'
    mappings, angle_rows, coverage, provenance, exposure = [], [], [], [], []
    for protein in config['enabled_proteins']:
        spec = config['proteins'][protein]
        for ref in spec['references']:
            rid = ref['id']
            path = raw / f'{rid}.cif'
            # Verify cache on every offline analysis too.
            retrieval = cached_download(f'https://files.rcsb.org/download/{rid}.cif', path, offline=True)
            metadata_path = raw / f'{rid}.metadata.json'
            metadata_provenance = cached_download(f'https://data.rcsb.org/rest/v1/core/entry/{rid}', metadata_path, offline=True)
            meta = json.loads(metadata_path.read_text())
            citation = meta['rcsb_primary_citation']
            if citation['pdbx_database_id_DOI'].lower() != ref['doi'].lower():
                raise ValueError(f'Reference citation mismatch for {rid}')
            residues, data = parse_cif(path)
            mapped = reference_mapping(residues, data, spec['segment'], spec['canonical_start'])
            chain_angles = angles(mapped, config['analysis']['peptide_bond_max_angstrom'])
            counts = {}
            for chain in sorted({r.label_chain for r in mapped}):
                aa = [a for a in chain_angles if a['label_chain'] == chain]
                counts[chain] = sum(np.isfinite(a['phi']) and np.isfinite(a['psi']) for a in aa)
            representative = sorted(counts, key=lambda c: (-counts[c], c))[0]
            for r in mapped:
                mappings.append({**r.identifiers(), 'protein': protein, 'reference': rid,
                                 'canonical_position': r.position,
                                 'precursor_position': r.position - spec['canonical_start'] + spec.get('precursor_start', spec['canonical_start']),
                                 'segment_position': r.position - spec['canonical_start'] + 1,
                                 'representative': r.label_chain == representative})
            for a in chain_angles:
                angle_rows.append({**a, 'protein': protein, 'reference': rid,
                                   'representative': a['label_chain'] == representative})
            for chain, valid in counts.items():
                positions = sorted(r.position for r in mapped if r.label_chain == chain)
                coverage.append({'protein': protein, 'reference': rid, 'role': ref['role'],
                                 'label_chain': chain, 'representative': chain == representative,
                                 'resolved_residues': len(positions), 'target_length': len(spec['segment']),
                                 'coordinate_coverage': len(positions) / len(spec['segment']),
                                 'resolved_positions': ';'.join(map(str, positions)), 'eligible_both_angles': valid})
            release = meta['rcsb_accession_info']['initial_release_date'][:10]
            provenance.append({'protein': protein, 'reference': rid, 'role': ref['role'],
                               'source_url': retrieval['url'], 'retrieved_utc': retrieval['retrieved_utc'],
                               'coordinate_sha256': retrieval['sha256'],
                               'metadata_sha256': metadata_provenance['sha256'], 'release_date': release,
                               'entry_title': meta['struct']['title'], 'citation_title': citation['title'],
                               'citation_year': citation['year'], 'doi': citation['pdbx_database_id_DOI'],
                               'representative_label_chain': representative})
            snapshot = config['model']['documented_pdb_snapshot']
            exposure.append({'protein': protein, 'reference': rid, 'release_date': release,
                             'configured_model_version': config['model']['version'],
                             'configured_checkpoint': config['model']['checkpoint'],
                             'documented_paper_pdb_snapshot': snapshot,
                             'predates_paper_snapshot': release <= snapshot,
                             'actual_checkpoint_training_cutoff': config['model']['training_cutoff'],
                             'training_inclusion': 'UNKNOWN; predating a snapshot is potential exposure, not proof of inclusion',
                             'template_setting_planned': config['model']['templates'],
                             'actual_template_exposure': 'UNKNOWN until inference provenance imported',
                             'unseen_generalization_claim': 'NOT SUPPORTED',
                             'documentation': config['model']['documentation']})
    processed = root / 'data/processed'
    table(processed / 'reference_mapping.csv', mappings)
    table(processed / 'reference_angles.csv', angle_rows)
    table(processed / 'reference_coverage.csv', coverage)
    table(processed / 'reference_provenance.csv', provenance)
    table(processed / 'exposure_audit.csv', exposure)
    return pd.DataFrame(angle_rows)


def reference_objects(config):
    """Load representative chains using the same deterministic policy as ingestion."""
    root = initialize(config)
    coverage = pd.read_csv(root / 'data/processed/reference_coverage.csv')
    out = {}
    for protein in config['enabled_proteins']:
        spec = config['proteins'][protein]
        for ref in spec['references']:
            rid = ref['id']
            chain = coverage.loc[(coverage.reference == rid) & coverage.representative, 'label_chain'].iloc[0]
            residues, data = parse_cif(root / f'data/raw/{rid}.cif')
            mapped = reference_mapping(residues, data, spec['segment'], spec['canonical_start'])
            out[rid] = [r for r in mapped if r.label_chain == chain]
    return out
