"""mmCIF residue identities, deterministic conformers, sequence mapping, torsions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1


def clean(value):
    return '' if value in ('.', '?', None) else str(value)


@dataclass
class Residue:
    label_chain: str
    auth_chain: str
    label_seq: int
    auth_seq: str
    insertion: str
    aa: str
    atoms: dict = field(default_factory=dict)
    atom_rows: dict = field(default_factory=dict)
    altloc: str = ''
    ambiguous: bool = False
    position: int | None = None

    def identifiers(self):
        return dict(label_chain=self.label_chain, auth_chain=self.auth_chain,
                    label_seq=self.label_seq, auth_seq=self.auth_seq,
                    insertion_code=self.insertion, amino_acid=self.aa, altloc=self.altloc)


def parse_cif(path):
    data = MMCIF2Dict(str(path))
    n = len(data.get('_atom_site.id', []))
    if not n:
        raise ValueError(f'No atom_site records in {path}')
    def col(name, default='?'):
        return data.get('_atom_site.' + name, [default] * n)
    columns = {k: col(k) for k in ['label_asym_id', 'auth_asym_id', 'label_seq_id', 'auth_seq_id',
                                  'pdbx_PDB_ins_code', 'label_comp_id', 'label_atom_id', 'label_alt_id',
                                  'Cartn_x', 'Cartn_y', 'Cartn_z', 'occupancy', 'B_iso_or_equiv']}
    models = col('pdbx_PDB_model_num', '1')
    if len(set(models)) != 1:
        raise ValueError(f'Multiple coordinate models in {path}; explicitly split models before importing')
    groups = {}
    for i in range(n):
        if not columns['label_seq_id'][i].isdigit():
            continue  # solvent/non-polymer
        key = (columns['label_asym_id'][i], int(columns['label_seq_id'][i]),
               clean(columns['pdbx_PDB_ins_code'][i]))
        groups.setdefault(key, []).append(i)
    residues = []
    for (chain, label, ins), indices in sorted(groups.items()):
        first = indices[0]
        variants = sorted({clean(columns['label_alt_id'][i]) for i in indices} - {''})
        def occupancy(alt):
            return sum(float(columns['occupancy'][i]) for i in indices
                       if clean(columns['label_alt_id'][i]) == alt and columns['label_atom_id'][i] in ('N', 'CA', 'C'))
        chosen = sorted(variants, key=lambda a: (-occupancy(a), a))[0] if variants else ''
        r = Residue(chain, columns['auth_asym_id'][first], label, columns['auth_seq_id'][first],
                    ins, seq1(columns['label_comp_id'][first], custom_map={'MSE': 'M'}), altloc=chosen)
        identities = {columns['label_comp_id'][i] for i in indices}
        r.ambiguous = len(identities) != 1 or bool(ins)
        for i in indices:
            alt = clean(columns['label_alt_id'][i])
            if alt not in ('', chosen):
                continue
            atom = columns['label_atom_id'][i]
            if atom in r.atoms:
                r.ambiguous = True
                continue
            r.atoms[atom] = np.array([float(columns[c][i]) for c in ('Cartn_x', 'Cartn_y', 'Cartn_z')])
            r.atom_rows[atom] = i
        residues.append(r)
    counts = {}
    for r in residues:
        counts[(r.label_chain, r.label_seq)] = counts.get((r.label_chain, r.label_seq), 0) + 1
    for r in residues:
        r.ambiguous |= counts[(r.label_chain, r.label_seq)] > 1
    return residues, data


def entity_sequences(data):
    seqs = {}
    for ent, num, mon in zip(data.get('_entity_poly_seq.entity_id', []),
                             data.get('_entity_poly_seq.num', []), data.get('_entity_poly_seq.mon_id', [])):
        seqs.setdefault(ent, {})[int(num)] = seq1(mon, custom_map={'MSE': 'M'})
    chains = dict(zip(data.get('_struct_asym.id', []), data.get('_struct_asym.entity_id', [])))
    return {chain: seqs[entity] for chain, entity in chains.items() if entity in seqs}


def reference_mapping(residues, data, segment, canonical_start):
    """Require unique exact sequence anchoring; do not guess author numbering."""
    entities = entity_sequences(data)
    mapped = []
    for chain in sorted({r.label_chain for r in residues}):
        chain_res = [r for r in residues if r.label_chain == chain]
        if chain not in entities:
            continue
        ent = entities[chain]
        labels = sorted(ent)
        if labels != list(range(1, len(labels) + 1)):
            raise ValueError(f'Nonconsecutive entity sequence labels in chain {chain}')
        seq = ''.join(ent[i] for i in labels)
        # References can contain the full native chain or only a resolved subsegment.
        candidates = []
        for offset in range(-len(seq) + 1, len(segment)):
            overlap = [(j, offset + j - 1) for j in labels if 0 <= offset + j - 1 < len(segment)]
            if len(overlap) >= min(10, len(segment), len(seq)) and all(ent[j] == segment[k] for j, k in overlap):
                # The shorter sequence must be wholly contained; reject arbitrary short local matches.
                if len(overlap) == min(len(seq), len(segment)):
                    candidates.append(offset)
        if not candidates:
            continue  # another protein in a heteromer
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous canonical sequence anchoring for reference chain {chain}')
        offset = candidates[0]
        for r in chain_res:
            k = offset + r.label_seq - 1
            if 0 <= k < len(segment):
                if r.aa != segment[k]:
                    raise ValueError(f'Reference sequence mismatch {chain}:{r.label_seq}: {r.aa} != {segment[k]}')
                r.position = canonical_start + k
                mapped.append(r)
    if not mapped:
        raise ValueError('No reference chain uniquely matches the configured canonical segment')
    return mapped


def map_prediction(residues, data, sequence, chain, label_offset=0):
    selected = [r for r in residues if r.label_chain == chain]
    if not selected:
        raise ValueError(f'Prediction label chain {chain!r} absent')
    entities = entity_sequences(data)
    if chain in entities:
        for label, aa in entities[chain].items():
            pos = label - label_offset
            if not 1 <= pos <= len(sequence) or sequence[pos - 1] != aa:
                raise ValueError(f'Prediction polymer sequence mismatch at {chain}:{label}; check label_offset')
    seen = set()
    for r in selected:
        pos = r.label_seq - label_offset
        if not 1 <= pos <= len(sequence):
            raise ValueError(f'Prediction residue {chain}:{r.label_seq} outside construct; check label_offset')
        if pos in seen:
            raise ValueError(f'Insertion/duplicate ambiguity at prediction {chain}:{r.label_seq}')
        seen.add(pos)
        if r.aa != sequence[pos - 1]:
            raise ValueError(f'Prediction sequence mismatch at construct {pos}: {r.aa} != {sequence[pos - 1]}')
        r.position = pos
    return selected


def dihedral(points):
    p = np.asarray(points, dtype=float)
    b0, b1, b2 = -(p[1] - p[0]), p[2] - p[1], p[3] - p[2]
    norm = np.linalg.norm(b1)
    if norm < 1e-10:
        return np.nan
    b1 = b1 / norm
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-10:
        return np.nan
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def angles(residues, bond_max=1.8):
    by_key = {(r.label_chain, r.position): r for r in residues if r.position is not None}
    rows = []
    def angle(r, step):
        neighbor = by_key.get((r.label_chain, r.position + step))
        if r.ambiguous or (neighbor and neighbor.ambiguous):
            return np.nan, 'insertion_or_identity_ambiguity'
        if not all(a in r.atoms for a in ('N', 'CA', 'C')):
            return np.nan, 'missing_backbone_atom'
        if neighbor is None:
            return np.nan, 'terminal_or_missing_neighbor'
        if neighbor.label_seq != r.label_seq + step:
            return np.nan, 'nonconsecutive_label_sequence'
        left, right = (neighbor, r) if step == -1 else (r, neighbor)
        if 'C' not in left.atoms or 'N' not in right.atoms:
            return np.nan, 'missing_neighbor_backbone_atom'
        distance = np.linalg.norm(left.atoms['C'] - right.atoms['N'])
        if not np.isfinite(distance) or distance > bond_max:
            return np.nan, 'peptide_chain_break'
        pts = [neighbor.atoms['C'], r.atoms['N'], r.atoms['CA'], r.atoms['C']] if step == -1 else [
            r.atoms['N'], r.atoms['CA'], r.atoms['C'], neighbor.atoms['N']]
        value = dihedral(pts)
        return value, '' if np.isfinite(value) else 'degenerate_geometry'
    for r in residues:
        if r.position is None:
            continue
        phi, pr = angle(r, -1)
        psi, sr = angle(r, 1)
        rows.append({**r.identifiers(), 'position': r.position, 'phi': phi, 'psi': psi,
                     'phi_missing_reason': pr, 'psi_missing_reason': sr})
    return rows
