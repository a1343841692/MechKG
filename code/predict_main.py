"""Evaluate frozen standard/hard test pairs with each supplied fold separately."""
import argparse
from pathlib import Path
import json
from release_runtime import ROOT, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/predictions')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--protocol', choices=['standard', 'hard', 'both'], default='standard')
    parser.add_argument('--fold', choices=['all', '1', '2', '3', '4', '5'], default='all')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--explain', action='store_true', help='Export per-pair channel contributions')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be a new or empty directory')
    runner, config = prepare(output, args.device)
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    from data_loader import load_knowledge_graph, load_named_splits, build_message_graph, assert_disjoint_splits
    from models import create_model
    from mechkg import configure_from_data
    from trainer import KFoldTrainer
    data = load_knowledge_graph(kg_file=runner.main.DATA_FILE).to(config.device)
    dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
    expected_dims = {'miRNA': 65, 'Gene': 65, 'Pathway': 768, 'Disease': 768}
    if dims != expected_dims:
        raise ValueError(f'Input features differ from the trained model: {dims}')
    folds, test = load_named_splits(data, config.data.fixed_split_file, 5)
    assert_disjoint_splits(folds, test)
    helper = KFoldTrainer.__new__(KFoldTrainer)
    helper.data, helper.device = data, config.device
    helper.pair_table = helper._load_pair_table()
    reverse = data.reverse_mapping
    standard = pd.DataFrame({'miRNA': [reverse['miRNA'][int(i)] for i in test['edges'][:, 0]],
                             'gene': [reverse['Gene'][int(i)] for i in test['edges'][:, 1]],
                             'label': test['labels'].cpu().numpy().astype(int)})
    sets = {'standard': (test['edges'], standard)}
    if args.protocol in ('hard', 'both'):
        hard = pd.read_csv(ROOT / 'data/splits/hard_test_pairs.csv')
        mi = {n.lower(): i for n, i in data.node_mapping['miRNA'].items()}
        gi = data.node_mapping['Gene']
        edges = torch.tensor([[mi[m.lower()], gi[g]] for m, g in zip(hard.miRNA, hard.gene)],
                             dtype=torch.long, device=config.device)
        sets['hard'] = edges, hard
    if args.protocol == 'hard':
        del sets['standard']
    wanted = range(1, 6) if args.fold == 'all' else [int(args.fold)]
    metrics = []
    with torch.no_grad():
        for k in wanted:
            tr, va = folds[k - 1]
            eval_sets = [va, test] + [{'edges': e} for name, (e, _) in sets.items() if name == 'hard']
            graph = build_message_graph(data.edge_index_dict, tr, eval_sets)
            checkpoint = torch.load(ROOT / f'models/fold_{k}_model.pth', map_location=config.device, weights_only=True)
            model = create_model(data.metadata(), dims, model_type='xgft').to(config.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            configure_from_data(model, data)
            model.eval()
            local, glob = model(data.x_dict, graph)
            for protocol, (edges, frame) in sets.items():
                probs, contributions = [], []
                for start in range(0, len(edges), args.batch_size):
                    batch = edges[start:start + args.batch_size]
                    pf = helper._pair(batch)
                    if args.explain:
                        p, parts = model.mechkg(local, graph, batch[:, 0], batch[:, 1], pair_features=pf, return_parts=True)
                        c = pd.DataFrame(parts['contrib'].cpu().numpy(), columns=[f'c_{n}' for n in model.mechkg.channel_names])
                        c['residual'] = parts['residual'].cpu().numpy()
                        c['bias'] = float(model.mechkg.bias)
                        contributions.append(c)
                    else:
                        p = model.predict_links(local, glob, batch[:, 0], batch[:, 1], pair_features=pf, graph=graph)
                    probs.append(p.cpu().numpy())
                p = np.concatenate(probs)
                result = frame.copy()
                result['fold'], result['probability'] = k, p
                if contributions:
                    result = pd.concat([result.reset_index(drop=True), pd.concat(contributions, ignore_index=True)], axis=1)
                result.to_csv(output / f'{protocol}_fold_{k}.csv', index=False)
                metrics.append({'protocol': protocol, 'fold': k, 'AUC': roc_auc_score(result.label, p),
                                'AP': average_precision_score(result.label, p)})
            del model, local, glob
            if config.device.type == 'cuda':
                torch.cuda.empty_cache()
    frame = pd.DataFrame(metrics)
    frame.to_csv(output / 'metrics_per_fold.csv', index=False)
    summary = {}
    for protocol, part in frame.groupby('protocol'):
        summary[protocol] = {name: {'mean': float(part[name].mean()),
                                   'sample_sd': float(part[name].std(ddof=1)) if len(part) > 1 else None}
                             for name in ['AUC', 'AP']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(frame.to_string(index=False))
    print('Saved:', output)


if __name__ == '__main__':
    main()
