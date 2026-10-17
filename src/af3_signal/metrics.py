"""Circular and intrachain geometry metrics with explicit empty-set semantics."""
import numpy as np


def circular_delta(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def angular_error(phi_pred, psi_pred, phi_ref, psi_ref):
    return (circular_delta(phi_pred, phi_ref) + circular_delta(psi_pred, psi_ref)) / 2


def mask_score(errors, mask, minimum):
    values = np.asarray(errors)[np.asarray(mask, dtype=bool)]
    if len(values) < minimum or not np.isfinite(values).all():
        return np.nan
    return float(values.mean())


def beta_fraction(phi, psi, phi_bounds=(-180, -90), psi_bounds=(90, 180)):
    phi, psi = np.asarray(phi), np.asarray(psi)
    valid = np.isfinite(phi) & np.isfinite(psi)
    if not valid.any():
        return np.nan
    return float(((phi[valid] >= phi_bounds[0]) & (phi[valid] <= phi_bounds[1]) &
                  (psi[valid] >= psi_bounds[0]) & (psi[valid] <= psi_bounds[1])).mean())


def rmsd(predicted, reference):
    p, r = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if len(p) < 3 or p.shape != r.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
        return np.nan
    p, r = p - p.mean(axis=0), r - r.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ r)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return float(np.sqrt(np.mean(np.sum((p @ rotation - r)**2, axis=1))))


def contacts(coords, distance=8., separation=3):
    positions = sorted(coords)
    return {(i, j) for k, i in enumerate(positions) for j in positions[k + 1:]
            if j - i > separation and np.linalg.norm(coords[i] - coords[j]) < distance}


def contact_overlap(predicted, reference):
    if not reference:
        return {'contact_recall': np.nan, 'contact_jaccard': np.nan, 'contact_status': 'no_eligible_reference_contacts'}
    union = predicted | reference
    return {'contact_recall': len(predicted & reference) / len(reference),
            'contact_jaccard': len(predicted & reference) / len(union), 'contact_status': 'OK'}
