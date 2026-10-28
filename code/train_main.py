"""Retrain the five main-model folds; supplied checkpoints are never overwritten."""
import argparse
from pathlib import Path
from release_runtime import ROOT, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/retrained_main')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be a new or empty directory')
    runner, _ = prepare(args.output, args.device)
    runner.main.main()


if __name__ == '__main__':
    main()
