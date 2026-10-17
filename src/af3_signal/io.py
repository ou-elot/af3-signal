"""Configuration, immutable raw cache, and portable tabular IO."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if 'extends' in config:
        parent = load_config(path.parent / config.pop('extends'))
        parent.update(config)
        config = parent
    config.setdefault('enabled_proteins', list(config['proteins']))
    if config['n_shuffles'] < 1 or len(set(config['prediction_seeds'])) != len(config['prediction_seeds']):
        raise ValueError('Need at least one shuffle and unique prediction seeds')
    if len({f['id'] for f in config['flanks']}) != len(config['flanks']):
        raise ValueError('Flank IDs must be unique')
    for name in config['enabled_proteins']:
        p = config['proteins'][name]
        if p['canonical_end'] - p['canonical_start'] + 1 != len(p['segment']):
            raise ValueError(f'{name}: canonical range length does not match segment sequence')
    a = config['analysis']
    if not 0 < a['min_coverage'] <= 1 or a['min_residues'] < 1:
        raise ValueError('Invalid analysis coverage/minimum residue thresholds')
    return config


def initialize(config):
    root = Path(config['run_dir'])
    for folder in ['data/raw', 'data/processed', 'inputs/af3', 'predictions', 'results', 'figures', 'reports']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    lock = root / 'config.lock.json'
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise ValueError(f'Configuration changed for {root}; choose a fresh run_dir to avoid mixing experiments')
    write_json(lock, config)
    return root


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, lineterminator='\n')
    return frame


def read_table(path):
    return pd.read_csv(path, keep_default_na=False)


def cached_download(url, path, offline=False):
    """Resume partial transfers when HTTP Range is honored; never replace committed raw files."""
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + '.provenance.json')
    if path.exists():
        if not meta_path.exists():
            raise ValueError(f'Untracked raw file: {path}; move it aside and fetch with provenance')
        meta = json.loads(meta_path.read_text())
        if meta['sha256'] != sha256(path) or meta['url'] != url:
            raise ValueError(f'Raw cache integrity/source mismatch: {path}')
        return meta
    if offline:
        raise ValueError(f'Offline cache missing: {path}; run fetch-references online first')
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    size = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={'User-Agent': 'af3-signal/0.1 research', **({'Range': f'bytes={size}-'} if size else {})})
    with urllib.request.urlopen(req, timeout=120) as response:
        resumed = size > 0 and response.status == 206
        if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {size}-'):
            raise ValueError(f'Invalid resume range from {url}')
        with part.open('ab' if resumed else 'wb') as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                handle.write(chunk)
        expected = response.headers.get('Content-Length')
        actual = part.stat().st_size - (size if resumed else 0)
        if expected and actual != int(expected):
            raise ValueError(f'Incomplete download: {url}; rerun to resume')
        headers = dict(response.headers)
    part.rename(path)
    meta = {'url': url, 'retrieved_utc': utcnow(), 'sha256': sha256(path), 'bytes': path.stat().st_size, 'http_headers': headers}
    write_json(meta_path, meta)
    return meta
