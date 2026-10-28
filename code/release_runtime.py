"""Load the frozen main-model configuration using paths inside this release."""
from pathlib import Path
import importlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, device='auto'):
    output = Path(output).resolve()
    for protected in ['code', 'data', 'models', 'results']:
        if output == ROOT / protected or ROOT / protected in output.parents:
            raise ValueError(f'Output must be outside the supplied {protected}/ directory')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.chdir(output)
    description = json.loads((ROOT / 'models/main_model_config.json').read_text(encoding='utf-8'))
    required = description['required_inputs']
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing release inputs: ' + ', '.join(missing))
    saved = sys.argv
    sys.argv = [str(ROOT / 'code/run_kg_v2.py'), *description['training_arguments'], '--run-dir', str(output)]
    try:
        runner = importlib.import_module('run_kg_v2')
    finally:
        sys.argv = saved
    from config import config, setup_experiment
    import torch
    if device != 'auto':
        config.device = torch.device(device)
    if config.device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu or install a CUDA-enabled PyTorch')
    setup_experiment('full_biological')
    return runner, config
