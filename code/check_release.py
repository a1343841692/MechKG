"""Verify files and stored results on CPU; does not train or run inference."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-only', action='store_true', help='Only require the Python standard library')
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    errors, checks = [], {}
    manifest = json.loads((ROOT / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest:
        path = ROOT / row['file']
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            errors.append('File missing or altered: ' + row['file'])
    checks['manifest_files'] = len(manifest)
    for path in (ROOT / 'code').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    weights = sorted(ROOT.rglob('*.pth'))
    assert len(weights) == 5 and all(p.parent == ROOT / 'models' for p in weights)
    checks['model_files'] = len(weights)
    if not args.hash_only:
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.metrics import roc_auc_score, average_precision_score
        from release_runtime import prepare
        with tempfile.TemporaryDirectory(prefix='mechkg_release_check_') as tmp:
            old = Path.cwd()
            try:
                _, config = prepare(Path(tmp), 'cpu')
                from models import create_model
                kg = pd.read_csv(ROOT / 'data/processed/kg_v2_xgft_coding_nosim.csv')
                assert len(kg) == 200224 and not kg.relation.str.contains('sequence_similar').any()
                iso = pd.read_csv(ROOT / 'data/processed/kg_v2_xgft_coding_nosim.isolated_nodes.csv')
                types = dict(zip(kg['head'], kg.head_type)) | dict(zip(kg['tail'], kg.tail_type)) | dict(zip(iso['name'], iso['type']))
                nodes = {nt: sorted(n for n, t in types.items() if t == nt) for nt in config.data.node_types}
                assert {nt: len(ns) for nt, ns in nodes.items()} == {'miRNA': 883, 'Gene': 3303, 'Pathway': 14071, 'Disease': 4199}
                edges = list(dict.fromkeys(zip(kg.head_type, kg.relation, kg.tail_type)))
                edges += [(b, 'reverse_' + r, a) for a, r, b in edges]
                splits = pd.read_csv(ROOT / 'data/splits/coding_named_splits.csv')
                valid_mi, valid_gi = {n.lower() for n in nodes['miRNA']}, set(nodes['Gene'])
                assert splits.miRNA.str.lower().isin(valid_mi).all() and splits.gene.isin(valid_gi).all()
                def pairs(df): return set(zip(df.miRNA.str.lower(), df.gene))
                test = splits[splits.split == 'test']
                assert len(test) == 3012
                heldout = pairs(test)
                for k in range(1, 6):
                    tr = splits[splits.split == f'fold{k}_train']
                    va = splits[splits.split == f'fold{k}_val']
                    assert len(pairs(tr)) == len(tr) and len(pairs(va)) == len(va)
                    assert not (pairs(tr) & pairs(va) or pairs(tr) & heldout or pairs(va) & heldout)
                hard = pd.read_csv(ROOT / 'data/splits/hard_test_pairs.csv')
                pos, neg = hard[hard.label == 1], hard[hard.label == 0]
                assert pairs(pos) == pairs(test[test.label == 1])
                for col in ['miRNA', 'gene']:
                    assert Counter(pos[col].str.lower()) == Counter(neg[col].str.lower())
                assert len(pairs(hard)) == len(hard) and not pairs(neg) & pairs(splits)
                known = kg[kg.relation == 'validated_interaction']
                assert not pairs(neg) & set(zip(known['head'].str.lower(), known['tail']))
                z = np.load(ROOT / 'data/processed/pair_seq_features_coding.npz')
                assert z['features'].shape == (883, 3303, 6) and np.isfinite(z['features']).all()
                assert {n.lower() for n in z['mirnas']} == valid_mi and set(z['genes']) == valid_gi
                dims = {'miRNA': 65, 'Gene': 65, 'Pathway': 768, 'Disease': 768}
                model = create_model((list(config.data.node_types), edges), dims)
                for p in weights:
                    ck = torch.load(p, map_location='cpu', weights_only=True)
                    model.load_state_dict(ck['model_state_dict'], strict=True)
                checks['strict_checkpoint_loads'] = 5
                stored = np.load(ROOT / 'results/hard_test/predictions.npz')
                measured = []
                for protocol in ['standard', 'hard']:
                    for k, p in enumerate(stored[protocol], start=1):
                        y = stored[protocol + '_labels']
                        measured.append({'protocol': protocol, 'fold': k, 'AUC': roc_auc_score(y, p), 'AP': average_precision_score(y, p)})
                frame = pd.DataFrame(measured)
                reference = pd.read_csv(ROOT / 'results/hard_test/hard_test_folds_nosim.csv')
                merged = frame.merge(reference, on=['protocol', 'fold'], suffixes=('', '_reference'))
                assert len(merged) == 10
                assert np.allclose(merged.AUC, merged.AUC_reference, atol=1e-12, rtol=0)
                assert np.allclose(merged.AP, merged.AP_reference, atol=1e-12, rtol=0)
                checks.update({'triples': len(kg), 'nodes': {nt: len(ns) for nt, ns in nodes.items()},
                               'independent_test_pairs': len(test), 'hard_test_pairs': len(hard),
                               'pair_feature_shape': list(z['features'].shape), 'split_disjointness': True,
                               'hard_endpoints_exactly_matched': True, 'stored_prediction_metrics_match': True,
                               'training_executed': False, 'inference_executed': False})
            finally:
                os.chdir(old)
    report = {'ok': not errors, 'checks': checks, 'errors': errors}
    if args.report:
        args.report.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
