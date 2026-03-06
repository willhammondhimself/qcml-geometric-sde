"""
One-command experiment pipeline orchestrator.

Runs the full experiment → register → validate → paper → verify chain.

Usage:
    python experiments/pipeline.py                    # Default mode
    python experiments/pipeline.py --mode quick       # Quick mode
    python experiments/pipeline.py --mode full        # Full mode
    python experiments/pipeline.py --skip-compile     # Skip paper compilation
    python experiments/pipeline.py --dry-run          # Show plan only
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


class PipelineStep:
    """A single pipeline step with timing and status tracking."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = 'pending'
        self.elapsed_s = 0.0
        self.message = ''

    def run(self, func, *args, **kwargs):
        """Execute the step, catching errors."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Step: {self.name} — {self.description}")
        logger.info('=' * 60)

        t0 = time.time()
        try:
            result = func(*args, **kwargs)
            self.status = 'passed'
            self.elapsed_s = time.time() - t0
            logger.info(f"  PASSED ({self.elapsed_s:.1f}s)")
            return result
        except Exception as e:
            self.status = 'failed'
            self.elapsed_s = time.time() - t0
            self.message = str(e)
            logger.error(f"  FAILED ({self.elapsed_s:.1f}s): {e}")
            return None


def step_run_experiments(mode: str) -> dict | None:
    """Run the experiment runner with the given mode."""
    from experiments.config import load_config
    from experiments.runner import ExperimentRunner

    cfg = load_config()
    runner = ExperimentRunner(cfg)
    output = runner.run_comparison(mode)
    return output


def step_register(output: dict, json_path: str | None) -> int | None:
    """Register experiment in SQLite registry."""
    from experiments.registry import ExperimentRegistry

    reg = ExperimentRegistry()
    try:
        exp_id = reg.register(output, json_path=json_path)
        logger.info(f"  Registered as experiment #{exp_id}")
        return exp_id
    finally:
        reg.close()


def step_validate(output: dict) -> list[dict]:
    """Validate experiment results."""
    from experiments.config import load_config
    from experiments.validate import validate_results

    cfg = load_config()
    issues = validate_results(output, cfg)

    n_critical = sum(1 for i in issues if i['severity'] == 'CRITICAL')
    n_warning = sum(1 for i in issues if i['severity'] == 'WARNING')
    n_info = sum(1 for i in issues if i['severity'] == 'INFO')

    logger.info(f"  Issues: {n_critical} critical, {n_warning} warnings, {n_info} info")

    if n_critical > 0:
        for i in issues:
            if i['severity'] == 'CRITICAL':
                logger.error(f"  CRITICAL: {i['message']}")
        raise RuntimeError(f"{n_critical} critical validation issues found")

    return issues


def step_populate_and_compile() -> None:
    """Run populate_paper.py --compile."""
    result = subprocess.run(
        [sys.executable, str(ROOT / 'paper' / 'populate_paper.py'), '--compile'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        logger.error(f"  stdout: {result.stdout[-500:]}")
        logger.error(f"  stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"populate_paper.py failed (exit {result.returncode})")


def step_verify_numbers() -> bool:
    """Run verify_paper_numbers.py."""
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'verify_paper_numbers.py')],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            logger.info(f"  {line}")
    if result.returncode != 0:
        raise RuntimeError("Paper number verification failed")
    return True


def run_pipeline(mode: str = 'default', skip_compile: bool = False,
                 dry_run: bool = False) -> int:
    """Execute the full pipeline.

    Args:
        mode: Experiment mode — 'default', 'quick', or 'full'.
        skip_compile: Skip paper compilation step.
        dry_run: Print plan and exit.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    steps = [
        PipelineStep('experiments', f'Run comparison ({mode})'),
        PipelineStep('register', 'Register in SQLite'),
        PipelineStep('validate', 'Validate results'),
    ]
    if not skip_compile:
        steps.append(PipelineStep('compile', 'Populate tables + compile PDF'))
        steps.append(PipelineStep('verify', 'Verify paper numbers'))

    if dry_run:
        print(f"\nPipeline plan (mode={mode}):")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s.name:15s} — {s.description}")
        print(f"\nSkip compile: {skip_compile}")
        return 0

    t_start = time.time()
    output = None
    json_path = None

    # Step 1: Run experiments
    output = steps[0].run(step_run_experiments, mode)
    if output is None:
        _print_dashboard(steps, time.time() - t_start)
        return 1

    # Extract JSON path from output
    out_dir = ROOT / 'experiments' / 'outputs' / 'regime_detection'
    jsons = sorted(out_dir.glob('causal_comparison_*.json'))
    if jsons:
        json_path = str(jsons[-1])

    # Step 2: Register
    steps[1].run(step_register, output, json_path)

    # Step 3: Validate
    result = steps[2].run(step_validate, output)
    if steps[2].status == 'failed':
        _print_dashboard(steps, time.time() - t_start)
        return 1

    # Step 4: Compile (optional)
    if not skip_compile:
        steps[3].run(step_populate_and_compile)
        if steps[3].status == 'failed':
            _print_dashboard(steps, time.time() - t_start)
            return 1

        # Step 5: Verify numbers
        steps[4].run(step_verify_numbers)

    _print_dashboard(steps, time.time() - t_start)

    failed = any(s.status == 'failed' for s in steps)
    return 1 if failed else 0


def _print_dashboard(steps: list[PipelineStep], total_s: float) -> None:
    """Print a summary dashboard."""
    print(f"\n{'=' * 60}")
    print(f"  Pipeline Dashboard")
    print(f"{'=' * 60}")

    for s in steps:
        if s.status == 'passed':
            icon = 'PASS'
        elif s.status == 'failed':
            icon = 'FAIL'
        else:
            icon = 'SKIP'

        msg = f"  {s.message}" if s.message else ""
        print(f"  {icon:4s}  {s.name:15s}  {s.elapsed_s:6.1f}s{msg}")

    print(f"{'─' * 60}")
    n_pass = sum(1 for s in steps if s.status == 'passed')
    n_fail = sum(1 for s in steps if s.status == 'failed')
    print(f"  Total: {n_pass} passed, {n_fail} failed, {total_s:.1f}s elapsed")
    print(f"{'=' * 60}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        force=True,
    )

    import argparse
    parser = argparse.ArgumentParser(description='Experiment pipeline orchestrator')
    parser.add_argument('--mode', default='default',
                        choices=['default', 'quick', 'full'],
                        help='Experiment mode (default: default)')
    parser.add_argument('--skip-compile', action='store_true',
                        help='Skip paper compilation')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show plan without executing')
    args = parser.parse_args()

    exit_code = run_pipeline(
        mode=args.mode,
        skip_compile=args.skip_compile,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
