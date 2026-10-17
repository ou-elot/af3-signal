import argparse
import sys

from .analysis import analyze
from .constructs import prepare_inputs
from .io import load_config
from .predictions import normalize_predictions
from .references import fetch_references
from .reporting import report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reproducible AF3/fibril descriptive pilot (manual prediction import)')
    parser.add_argument('command', choices=['fetch-references', 'prepare-inputs', 'validate-predictions', 'analyze', 'report'])
    parser.add_argument('--config', default='configs/pilot.yaml')
    parser.add_argument('--manifest', help='CSV; file paths inside are relative to the configured run_dir')
    parser.add_argument('--offline', action='store_true', help='Use hash-verified raw cache only')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == 'fetch-references':
            frame = fetch_references(config, args.offline)
            print(f'Parsed {len(frame)} reference residue records; representative chains selected deterministically.')
        elif args.command == 'prepare-inputs':
            count, jobs = prepare_inputs(config)
            print(f'Prepared {count} constructs and {jobs} construct-seed jobs (PENDING inference).')
        elif args.command == 'validate-predictions':
            normalized, _, _ = normalize_predictions(config, args.manifest)
            print(f'Validated {len(normalized)} actual output samples; see normalized manifest and pending_jobs.csv.')
        elif args.command == 'analyze':
            frame = analyze(config, args.manifest)
            print(f'Analyzed {len(frame)} sample/reference comparisons.' if len(frame) else 'BLOCKED/PENDING: no AF3 predictions imported; reference and empty result tables written.')
        else:
            print(f'Report written: {report(config)}')
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return_code = 2
        raise SystemExit(return_code) from exc
