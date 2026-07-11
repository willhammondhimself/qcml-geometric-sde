"""Engineering layer for the honest nested-HPO pipeline: memoization, resumable
Optuna studies, and process-parallel execution.

The nested protocol re-evaluates the same (detector, params, crisis) Cohen's d
many times — selection pools are strict prefixes across expanding windows, and
Optuna revisits overlapping configs. Memoizing that float is the dominant
speedup (hours → minutes). Cache key reuses the runner.py cell-cache pattern
(sha256 over class+params+crisis+window+lib_hash+data_hash) so it invalidates on
a library change or a data change.

Nothing here runs unless explicitly installed; importing this module has no
effect on the default (uncached) pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from experiments.data_loader import ALL_CRISES
from experiments.runner import _get_library_hash

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "experiments" / "outputs" / "cache" / "cohens_d"


def data_fingerprint(X_enriched: np.ndarray) -> str:
    """Stable 12-hex fingerprint of the enriched feature matrix (cache invalidation)."""
    return hashlib.sha256(np.ascontiguousarray(X_enriched, dtype=np.float64).tobytes()).hexdigest()[
        :12
    ]


class CohensDCache:
    """Memoizes (detector_class, full-params, crisis_key, window_size) → Cohen's d.

    Two tiers: an in-process dict and a pickle file per key. Install() rebinds
    ``walk_forward_hpo._D_FN`` to ``get_or_compute`` so the existing protocol
    functions transparently hit the cache.
    """

    def __init__(self, cache_dir: Path | None = None, force: bool = False, compute_fn=None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self._compute_fn = compute_fn  # resolved to walk_forward_hpo._compute_d on install()
        self._lib_hash = _get_library_hash()
        self._data_hash = ""
        self._mem: dict[str, float] = {}
        self.hits = 0
        self.misses = 0

    def attach_data(self, X_enriched: np.ndarray) -> "CohensDCache":
        self._data_hash = data_fingerprint(X_enriched)
        return self

    def key(self, detector_class, params: dict, crisis_key: str, window_size: int) -> str:
        payload = json.dumps(
            {
                "method": getattr(detector_class, "__name__", str(detector_class)),
                "params": params,
                "crisis": crisis_key,
                "crisis_def": ALL_CRISES.get(crisis_key, {}),
                "window_size": window_size,
                "lib_hash": self._lib_hash,
                "data_hash": self._data_hash,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get_or_compute(
        self, detector_class, params, X_enriched, dates_enriched, crisis_key, window_size=10
    ):
        k = self.key(detector_class, params, crisis_key, window_size)
        if k in self._mem:
            self.hits += 1
            return self._mem[k]

        path = self.cache_dir / f"{k}.pkl"
        if not self.force and path.exists():
            try:
                with open(path, "rb") as f:
                    val = pickle.load(f)
                self._mem[k] = val
                self.hits += 1
                return val
            except Exception:
                pass  # corrupt cache entry → recompute

        self.misses += 1
        fn = self._compute_fn or self._resolve_default()
        val = fn(detector_class, params, X_enriched, dates_enriched, crisis_key, window_size)
        self._mem[k] = val
        # Atomic write so concurrent workers never read a half-written file
        # (identical value → last-writer-wins is safe).
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(val, f)
        os.replace(tmp, path)
        return val

    @staticmethod
    def _resolve_default():
        import experiments.walk_forward_hpo as wf

        return wf._compute_d

    def install(self) -> "CohensDCache":
        """Rebind walk_forward_hpo._D_FN to this cache (idempotent)."""
        import experiments.walk_forward_hpo as wf

        if self._compute_fn is None:
            self._compute_fn = wf._compute_d
        wf._D_FN = self.get_or_compute
        return self

    @staticmethod
    def uninstall():
        import experiments.walk_forward_hpo as wf

        wf._D_FN = wf._compute_d
