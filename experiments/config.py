"""
Configuration loader for experiment parameters.

Loads experiments/config.yaml and provides typed access to all experiment
settings: data sources, crisis definitions, detector configs, baselines,
and evaluation parameters.

Usage:
    from experiments.config import load_config, resolve_experiment

    cfg = load_config()
    methods, crises = resolve_experiment(cfg, 'default')
"""

import importlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'experiments' / 'config.yaml'


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and return the experiment configuration dictionary.

    Args:
        path: Path to config.yaml. Defaults to experiments/config.yaml.

    Returns:
        Parsed YAML as a nested dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_class(class_path: str) -> type:
    """Import and return a class from a dotted path.

    Args:
        class_path: e.g. 'qcml_geometry.BerryPhaseRateDetector'

    Returns:
        The class object.
    """
    parts = class_path.rsplit('.', 1)
    module_path, class_name = parts[0], parts[1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_detector_configs(cfg: dict) -> dict[str, dict[str, Any]]:
    """Build detector config dicts compatible with regime_comparison.py.

    Returns:
        Dict mapping display_name -> {'class': <class>, 'params': {...}}
    """
    result = {}
    for key, det_cfg in cfg['detectors'].items():
        result[det_cfg['display_name']] = {
            'class': resolve_class(det_cfg['class']),
            'params': dict(det_cfg['params']),
        }
    return result


def get_baseline_configs(cfg: dict) -> dict[str, dict[str, Any]]:
    """Build baseline config dicts compatible with regime_comparison.py.

    Returns:
        Dict mapping display_name -> {'class': <class>, 'params': {...}, 'supervised': bool}
    """
    result = {}
    for key, bl_cfg in cfg['baselines'].items():
        result[bl_cfg['display_name']] = {
            'class': resolve_class(bl_cfg['class']),
            'params': dict(bl_cfg['params']),
            'supervised': bl_cfg.get('supervised', False),
            'evaluation': bl_cfg.get('evaluation'),
        }
    return result


def get_crises(cfg: dict, subset: str = 'post_2005') -> dict[str, dict]:
    """Return crisis definitions for a named subset.

    Args:
        cfg: Loaded config dict.
        subset: Key from crisis_subsets (quick, post_2005, all, novel, conventional).

    Returns:
        Dict of crisis_key -> {start, end, label, category}.
    """
    keys = cfg['crisis_subsets'][subset]
    return {k: cfg['crises'][k] for k in keys if k in cfg['crises']}


def resolve_experiment(cfg: dict, experiment_name: str = 'default') -> tuple[list[str], dict]:
    """Resolve a named experiment into method keys and crisis definitions.

    Args:
        cfg: Loaded config dict.
        experiment_name: Key from experiments section.

    Returns:
        (method_keys, crises_dict) where method_keys are config keys
        and crises_dict maps crisis_key -> {start, end, label}.
    """
    exp = cfg['experiments'][experiment_name]
    method_keys = exp['methods']
    crises = get_crises(cfg, exp['crisis_subset'])
    return method_keys, crises


def get_method_config(cfg: dict, method_key: str) -> dict[str, Any]:
    """Get class and params for a single method (detector or baseline).

    Args:
        cfg: Loaded config dict.
        method_key: Key from detectors or baselines section.

    Returns:
        {'class': <class>, 'params': {...}, 'display_name': str, 'supervised': bool}
    """
    if method_key in cfg['detectors']:
        det = cfg['detectors'][method_key]
        return {
            'class': resolve_class(det['class']),
            'params': dict(det['params']),
            'display_name': det['display_name'],
            'supervised': False,
        }
    elif method_key in cfg.get('info_geometry', {}):
        ig = cfg['info_geometry'][method_key]
        return {
            'class': resolve_class(ig['class']),
            'params': dict(ig['params']),
            'display_name': ig['display_name'],
            'supervised': False,
        }
    elif method_key in cfg['baselines']:
        bl = cfg['baselines'][method_key]
        return {
            'class': resolve_class(bl['class']),
            'params': dict(bl['params']),
            'display_name': bl['display_name'],
            'supervised': bl.get('supervised', False),
            'evaluation': bl.get('evaluation'),
        }
    else:
        raise KeyError(f"Unknown method key: {method_key}")


def config_hash(cfg: dict) -> str:
    """Compute a stable hash of the config for cache invalidation.

    Returns:
        Hex string (first 12 chars of SHA256).
    """
    import hashlib
    import json
    content = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:12]
