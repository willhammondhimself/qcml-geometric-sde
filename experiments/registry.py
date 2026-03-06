"""
SQLite experiment registry for tracking runs, results, and artifacts.

Provides full provenance: every experiment gets a unique ID, config hash,
git commit, timing, and status. Results are stored per (method, crisis) cell
with d-values and confidence intervals.

Tables:
    experiments — id, name, config_hash, git_commit, timing, status
    results     — experiment_id, method, crisis, d_value, ci_lo, ci_hi
    artifacts   — experiment_id, type, path

CLI:
    python -m experiments.registry list
    python -m experiments.registry latest
    python -m experiments.registry query --method "Berry Phase Rate"
    python -m experiments.registry diff <id1> <id2>
    python -m experiments.registry backfill <json_path>
"""

import json
import logging
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / 'experiments' / 'outputs' / 'registry.db'


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 'unknown'


class ExperimentRegistry:
    """SQLite-backed experiment registry.

    Args:
        db_path: Path to SQLite database. Created if missing.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                config_hash TEXT,
                git_commit TEXT,
                timestamp TEXT NOT NULL,
                elapsed_s REAL,
                n_methods INTEGER,
                n_crises INTEGER,
                cache_hits INTEGER,
                total_cells INTEGER,
                status TEXT DEFAULT 'completed',
                json_path TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                crisis TEXT NOT NULL,
                d_value REAL,
                ci_lo REAL,
                ci_hi REAL,
                timing_s REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                UNIQUE(experiment_id, method, crisis)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                type TEXT NOT NULL,
                path TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_results_method ON results(method);
            CREATE INDEX IF NOT EXISTS idx_results_crisis ON results(crisis);
            CREATE INDEX IF NOT EXISTS idx_results_experiment ON results(experiment_id);

            CREATE TABLE IF NOT EXISTS canonical (
                name TEXT PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                json_path TEXT NOT NULL,
                set_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
        """)
        self.conn.commit()

    def register(self, output: dict, json_path: Optional[str] = None) -> int:
        """Register a completed experiment run.

        Args:
            output: Results dict from ExperimentRunner.run_comparison().
            json_path: Path to the saved JSON file.

        Returns:
            Experiment ID.
        """
        config = output.get('config', {})
        exp_name = output.get('experiment', 'unknown')

        cur = self.conn.execute("""
            INSERT INTO experiments (name, config_hash, git_commit, timestamp,
                                    elapsed_s, n_methods, n_crises,
                                    cache_hits, total_cells, status, json_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?)
        """, (
            exp_name,
            config.get('lib_hash', ''),
            _get_git_commit(),
            output.get('timestamp', datetime.now().isoformat()),
            config.get('elapsed_s'),
            config.get('n_methods'),
            config.get('n_crises'),
            config.get('cache_hits'),
            config.get('total_cells'),
            json_path,
        ))
        exp_id = cur.lastrowid

        # Insert per-cell results
        for method, crises in output.get('results', {}).items():
            for crisis, cell in crises.items():
                if isinstance(cell, dict):
                    self.conn.execute("""
                        INSERT OR REPLACE INTO results
                            (experiment_id, method, crisis, d_value, ci_lo, ci_hi, timing_s)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        exp_id, method, crisis,
                        cell.get('d'), cell.get('ci_lo'), cell.get('ci_hi'),
                        cell.get('timing_s'),
                    ))

        # Register JSON as artifact
        if json_path:
            self.conn.execute("""
                INSERT INTO artifacts (experiment_id, type, path)
                VALUES (?, 'json', ?)
            """, (exp_id, json_path))

        self.conn.commit()
        logger.info(f"Registered experiment #{exp_id}: {exp_name}")
        return exp_id

    def list_experiments(self, limit: int = 20) -> list[dict]:
        """List recent experiments."""
        rows = self.conn.execute("""
            SELECT id, name, timestamp, n_methods, n_crises, elapsed_s,
                   cache_hits, total_cells, git_commit, status
            FROM experiments
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def latest(self, name: Optional[str] = None) -> Optional[dict]:
        """Get the latest experiment, optionally filtered by name."""
        if name:
            row = self.conn.execute("""
                SELECT * FROM experiments WHERE name = ?
                ORDER BY id DESC LIMIT 1
            """, (name,)).fetchone()
        else:
            row = self.conn.execute("""
                SELECT * FROM experiments ORDER BY id DESC LIMIT 1
            """).fetchone()
        return dict(row) if row else None

    def get_results(self, experiment_id: int) -> list[dict]:
        """Get all results for an experiment."""
        rows = self.conn.execute("""
            SELECT method, crisis, d_value, ci_lo, ci_hi, timing_s
            FROM results WHERE experiment_id = ?
            ORDER BY method, crisis
        """, (experiment_id,)).fetchall()
        return [dict(r) for r in rows]

    def query_method(self, method: str, limit: int = 10) -> list[dict]:
        """Query results across experiments for a specific method."""
        rows = self.conn.execute("""
            SELECT e.id as experiment_id, e.name, e.timestamp,
                   r.crisis, r.d_value, r.ci_lo, r.ci_hi
            FROM results r
            JOIN experiments e ON r.experiment_id = e.id
            WHERE r.method = ?
            ORDER BY e.id DESC, r.crisis
            LIMIT ?
        """, (method, limit * 16)).fetchall()
        return [dict(r) for r in rows]

    def diff(self, id1: int, id2: int) -> dict:
        """Compare results between two experiments.

        Returns:
            Dict with per-method d-value changes.
        """
        r1 = {(r['method'], r['crisis']): r['d_value']
               for r in self.get_results(id1)}
        r2 = {(r['method'], r['crisis']): r['d_value']
               for r in self.get_results(id2)}

        all_keys = sorted(set(r1.keys()) | set(r2.keys()))
        changes = []
        for key in all_keys:
            d1 = r1.get(key)
            d2 = r2.get(key)
            if d1 is not None and d2 is not None:
                delta = d2 - d1
                changes.append({
                    'method': key[0], 'crisis': key[1],
                    'd_old': d1, 'd_new': d2, 'delta': delta,
                })
            elif d1 is None and d2 is not None:
                changes.append({
                    'method': key[0], 'crisis': key[1],
                    'd_old': None, 'd_new': d2, 'delta': None,
                })

        return {
            'experiment_1': id1,
            'experiment_2': id2,
            'n_cells': len(changes),
            'changes': sorted(changes, key=lambda x: abs(x.get('delta') or 0), reverse=True),
        }

    def backfill(self, json_path: str):
        """Import results from an existing timestamped JSON file.

        Reads the standard comparison JSON format and registers it.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")

        with open(path) as f:
            data = json.load(f)

        exp_id = self.register(data, json_path=str(path))
        logger.info(f"Backfilled {path.name} as experiment #{exp_id}")
        return exp_id

    # ── Canonical JSON tracking ──────────────────────────────────────

    def canonical_set(self, name: str, experiment_id: int) -> None:
        """Set a canonical JSON reference.

        Args:
            name: Canonical name (e.g. 'main_comparison', 'fusion_full').
            experiment_id: Experiment ID to mark as canonical.
        """
        exp = self.conn.execute(
            "SELECT json_path FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if not exp:
            raise ValueError(f"Experiment #{experiment_id} not found")

        json_path = exp['json_path'] or ''
        self.conn.execute("""
            INSERT OR REPLACE INTO canonical (name, experiment_id, json_path, set_at)
            VALUES (?, ?, ?, ?)
        """, (name, experiment_id, json_path, datetime.now().isoformat()))
        self.conn.commit()
        logger.info(f"Canonical '{name}' set to experiment #{experiment_id} ({json_path})")

    def canonical_list(self) -> list[dict]:
        """List all canonical JSON references."""
        rows = self.conn.execute("""
            SELECT c.name, c.experiment_id, c.json_path, c.set_at,
                   e.timestamp as exp_timestamp, e.n_methods, e.n_crises
            FROM canonical c
            JOIN experiments e ON c.experiment_id = e.id
            ORDER BY c.name
        """).fetchall()
        return [dict(r) for r in rows]

    def canonical_get(self, name: str) -> dict | None:
        """Get a specific canonical reference.

        Args:
            name: Canonical name.

        Returns:
            Dict with canonical info, or None if not set.
        """
        row = self.conn.execute("""
            SELECT c.name, c.experiment_id, c.json_path, c.set_at,
                   e.timestamp as exp_timestamp, e.n_methods, e.n_crises
            FROM canonical c
            JOIN experiments e ON c.experiment_id = e.id
            WHERE c.name = ?
        """, (name,)).fetchone()
        return dict(row) if row else None

    def close(self):
        self.conn.close()


def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Experiment registry CLI')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('list', help='List recent experiments')
    sub.add_parser('latest', help='Show latest experiment')

    q = sub.add_parser('query', help='Query results for a method')
    q.add_argument('--method', required=True, help='Method name')

    d = sub.add_parser('diff', help='Compare two experiments')
    d.add_argument('id1', type=int)
    d.add_argument('id2', type=int)

    b = sub.add_parser('backfill', help='Import existing JSON results')
    b.add_argument('json_path', help='Path to comparison JSON')

    ba = sub.add_parser('backfill-all', help='Import all existing JSON results')

    # Canonical subcommands
    cs = sub.add_parser('canonical', help='Manage canonical JSON references')
    cs_sub = cs.add_subparsers(dest='canonical_command')
    cs_set = cs_sub.add_parser('set', help='Set a canonical reference')
    cs_set.add_argument('canonical_name', help='Name (main_comparison, fusion_full, etc.)')
    cs_set.add_argument('exp_id', type=int, help='Experiment ID')
    cs_sub.add_parser('list', help='List canonical references')
    cs_get = cs_sub.add_parser('get', help='Get a canonical reference')
    cs_get.add_argument('canonical_name', help='Canonical name')

    args = parser.parse_args()
    reg = ExperimentRegistry()

    if args.command == 'list':
        for exp in reg.list_experiments():
            print(f"  #{exp['id']:3d}  {exp['name']:15s}  "
                  f"{exp['timestamp'][:19]}  "
                  f"{exp['n_methods']}m x {exp['n_crises']}c  "
                  f"{exp['elapsed_s'] or '?':>6}s  "
                  f"({exp['cache_hits'] or 0}/{exp['total_cells'] or '?'} cached)")

    elif args.command == 'latest':
        exp = reg.latest()
        if exp:
            print(json.dumps(exp, indent=2, default=str))
        else:
            print("No experiments registered yet.")

    elif args.command == 'query':
        results = reg.query_method(args.method)
        for r in results:
            d = r['d_value']
            d_str = f"{d:.3f}" if d is not None else "N/A"
            print(f"  exp#{r['experiment_id']:3d}  {r['crisis']:20s}  d={d_str}")

    elif args.command == 'diff':
        diff = reg.diff(args.id1, args.id2)
        print(f"Comparing experiment #{args.id1} vs #{args.id2} ({diff['n_cells']} cells):\n")
        for c in diff['changes'][:20]:
            d_old = f"{c['d_old']:.3f}" if c['d_old'] is not None else "N/A"
            d_new = f"{c['d_new']:.3f}" if c['d_new'] is not None else "N/A"
            delta = f"{c['delta']:+.3f}" if c['delta'] is not None else "new"
            print(f"  {c['method']:25s}  {c['crisis']:20s}  {d_old} -> {d_new}  ({delta})")

    elif args.command == 'backfill':
        reg.backfill(args.json_path)

    elif args.command == 'backfill-all':
        out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
        jsons = sorted(out_dir.glob('*comparison*.json'))
        for jp in jsons:
            try:
                reg.backfill(str(jp))
            except Exception as e:
                logger.warning(f"  Failed to backfill {jp.name}: {e}")
        print(f"Backfilled {len(jsons)} JSON files.")

    elif args.command == 'canonical':
        if args.canonical_command == 'set':
            reg.canonical_set(args.canonical_name, args.exp_id)
        elif args.canonical_command == 'list':
            entries = reg.canonical_list()
            if not entries:
                print("No canonical references set.")
            else:
                for e in entries:
                    print(f"  {e['name']:20s}  exp#{e['experiment_id']:3d}  "
                          f"{e['n_methods']}m x {e['n_crises']}c  "
                          f"set {e['set_at'][:10]}  "
                          f"{Path(e['json_path']).name if e['json_path'] else 'no path'}")
        elif args.canonical_command == 'get':
            entry = reg.canonical_get(args.canonical_name)
            if entry:
                print(json.dumps(entry, indent=2, default=str))
            else:
                print(f"Canonical '{args.canonical_name}' not set.")
        else:
            print("Usage: canonical {set|list|get}")

    else:
        parser.print_help()

    reg.close()


if __name__ == '__main__':
    main()
