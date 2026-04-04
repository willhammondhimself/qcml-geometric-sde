"""
Pre-submission quality gate with 8 checks.

Gates:
    1. Tests (pytest)
    2. Lint (ruff)
    3. Paper Numbers (verify_paper_numbers.py)
    4. Paper Compile (pdflatex x 3 + bibtex)
    5. Citation Verification (verify_citations.py)
    6. Canonical JSON Currency (is canonical JSON from latest run?)
    7. Registry Completeness (all paper tables covered?)
    8. Review Response Coverage (all review concerns addressed?)

Usage:
    python scripts/pre_submit_gate.py
    python scripts/pre_submit_gate.py --skip-compile
    python scripts/pre_submit_gate.py --gates 1,2,3
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class Gate:
    """A single quality gate."""

    def __init__(self, number: int, name: str, description: str):
        self.number = number
        self.name = name
        self.description = description
        self.status = 'SKIP'
        self.message = ''
        self.details = []

    def pass_(self, message: str = ''):
        self.status = 'PASS'
        self.message = message

    def warn(self, message: str = ''):
        self.status = 'WARN'
        self.message = message

    def fail(self, message: str = ''):
        self.status = 'FAIL'
        self.message = message


def _run_cmd(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a command, capturing output."""
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def gate_tests(g: Gate) -> None:
    """Gate 1: Run pytest."""
    result = _run_cmd([
        sys.executable, '-m', 'pytest', 'tests/', '-v',
        '--ignore=tests/test_crisis_validation.py', '-x', '-q',
    ])
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        summary = lines[-1] if lines else 'passed'
        g.pass_(summary)
    else:
        last_lines = result.stdout.strip().split('\n')[-3:]
        g.fail('\n'.join(last_lines))


def gate_lint(g: Gate) -> None:
    """Gate 2: Run ruff (CI scope: qcml_geometry/ and tests/ only)."""
    result = _run_cmd([
        sys.executable, '-m', 'ruff', 'check',
        'qcml_geometry/', 'tests/',
    ])
    if result.returncode == 0:
        g.pass_('No lint issues')
    else:
        n_issues = result.stdout.count('\n')
        g.fail(f'{n_issues} lint issues')
        g.details = result.stdout.strip().split('\n')[:5]


def gate_paper_numbers(g: Gate) -> None:
    """Gate 3: Verify paper numbers match source data."""
    script = REPO_ROOT / 'scripts' / 'verify_paper_numbers.py'
    if not script.exists():
        g.warn('verify_paper_numbers.py not found')
        return

    result = _run_cmd([sys.executable, str(script)])
    if result.returncode == 0:
        g.pass_(result.stdout.strip().split('\n')[-1] if result.stdout else 'OK')
    else:
        g.fail(result.stdout.strip().split('\n')[-1] if result.stdout else 'Failed')
        g.details = result.stdout.strip().split('\n')[-5:]


def gate_paper_compile(g: Gate) -> None:
    """Gate 4: Compile paper with pdflatex x 3 + bibtex."""
    paper_dir = REPO_ROOT / 'paper'
    cmds = [
        ['pdflatex', '-interaction=nonstopmode', 'qcml_geometric_sde.tex'],
        ['bibtex', 'qcml_geometric_sde'],
        ['pdflatex', '-interaction=nonstopmode', 'qcml_geometric_sde.tex'],
        ['pdflatex', '-interaction=nonstopmode', 'qcml_geometric_sde.tex'],
    ]
    for cmd in cmds:
        result = subprocess.run(
            cmd, cwd=paper_dir, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 and cmd[0] == 'pdflatex':
            g.fail(f'{" ".join(cmd)} failed')
            # Extract error from log
            log = paper_dir / 'qcml_geometric_sde.log'
            if log.exists():
                lines = log.read_text().split('\n')
                errors = [l for l in lines if l.startswith('!')]
                g.details = errors[:3]
            return

    pdf = paper_dir / 'qcml_geometric_sde.pdf'
    if pdf.exists():
        size_kb = pdf.stat().st_size / 1024
        g.pass_(f'PDF generated ({size_kb:.0f} KB)')
    else:
        g.fail('PDF not generated')


def gate_citations(g: Gate) -> None:
    """Gate 5: Verify citations."""
    script = REPO_ROOT / 'paper' / 'verify_citations.py'
    if not script.exists():
        g.warn('verify_citations.py not found')
        return

    result = _run_cmd([sys.executable, str(script), '--dry-run'], timeout=120)
    if result.returncode == 0:
        g.pass_('Citations verified')
    else:
        g.warn('Citation issues found')
        g.details = result.stdout.strip().split('\n')[-3:]


def gate_canonical_currency(g: Gate) -> None:
    """Gate 6: Check if canonical JSON is from the latest experiment run."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from experiments.registry import ExperimentRegistry
        reg = ExperimentRegistry()

        canonical = reg.canonical_list()
        latest = reg.latest()

        if not canonical:
            g.warn('No canonical JSON references set')
            reg.close()
            return

        if not latest:
            g.warn('No experiments in registry')
            reg.close()
            return

        stale = []
        for c in canonical:
            if c['experiment_id'] != latest['id']:
                stale.append(f"{c['name']} (exp#{c['experiment_id']}, latest=#{latest['id']})")

        if stale:
            g.warn(f'{len(stale)} stale canonical refs: {", ".join(stale)}')
        else:
            g.pass_('All canonical refs point to latest experiment')

        reg.close()
    except Exception as e:
        g.warn(f'Could not check: {e}')


def gate_registry_completeness(g: Gate) -> None:
    """Gate 7: Check results_registry.yaml covers all paper tables."""
    registry_path = REPO_ROOT / 'memory' / 'results_registry.yaml'
    if not registry_path.exists():
        g.warn('results_registry.yaml not found')
        return

    try:
        import yaml
        with open(registry_path) as f:
            registry = yaml.safe_load(f)
    except ImportError:
        g.warn('PyYAML not installed, skipping')
        return

    if not registry:
        g.warn('Empty registry')
        return

    claims = registry.get('claims', [])
    if not claims:
        g.warn('No claims in registry')
        return

    verified = sum(1 for c in claims if c.get('verified', False))
    total = len(claims)

    if verified == total:
        g.pass_(f'{verified}/{total} claims tracked')
    elif verified / total >= 0.9:
        g.warn(f'{verified}/{total} claims tracked ({total - verified} unverified)')
    else:
        g.fail(f'Only {verified}/{total} claims tracked')


def gate_review_coverage(g: Gate) -> None:
    """Gate 8: Check if review concerns are addressed.

    If an issue_registry.yaml exists, checks individual issue resolution.
    Falls back to string-counting heuristic otherwise.
    """
    registry_path = REPO_ROOT / 'paper' / 'review' / 'issue_registry.yaml'

    # Prefer structured registry if available
    if registry_path.exists():
        try:
            import yaml
            with open(registry_path) as f:
                registry = yaml.safe_load(f) or {}
        except ImportError:
            registry = None

        if registry and registry.get('issues'):
            issues = registry['issues']
            by_sev = {}
            for issue in issues:
                sev = issue.get('severity', 'MINOR').upper()
                by_sev.setdefault(sev, []).append(issue)

            # Check CRITICAL items
            critical = by_sev.get('CRITICAL', [])
            resolved_statuses = ('fixed', 'verified', 'wontfix')
            critical_resolved = sum(
                1 for i in critical
                if i.get('status') in resolved_statuses
            )
            critical_total = len(critical)

            # Check MAJOR items
            major = by_sev.get('MAJOR', [])
            major_resolved = sum(
                1 for i in major
                if i.get('status') in (*resolved_statuses, 'deferred')
            )
            major_total = len(major)

            detail = (f'{critical_resolved}/{critical_total} CRITICAL resolved, '
                      f'{major_resolved}/{major_total} MAJOR resolved')
            g.details.append(detail)

            if critical_resolved < critical_total:
                unresolved = [i['id'] for i in critical
                              if i.get('status') not in resolved_statuses]
                g.fail(f'Unresolved CRITICAL: {", ".join(unresolved)}')
            elif major_resolved < major_total:
                unresolved = [i['id'] for i in major
                              if i.get('status') not in (*resolved_statuses, 'deferred')]
                g.warn(f'Unresolved MAJOR: {", ".join(unresolved)}')
            else:
                g.pass_(f'All {critical_total} CRITICAL + {major_total} MAJOR resolved')
            return

    # Fallback: string-counting heuristic (no registry)
    reviews_dir = REPO_ROOT / 'paper' / 'review' / 'reviews'
    response_file = REPO_ROOT / 'paper' / 'response_to_reviewers.md'

    if not reviews_dir.exists():
        g.warn('No reviews directory')
        return

    syntheses = sorted(reviews_dir.glob('synthesis_*.md'))
    reviews = sorted(reviews_dir.glob('review_*.md'))

    if not reviews:
        g.warn('No review files found')
        return

    if syntheses:
        latest = syntheses[-1]
        content = latest.read_text()
        n_critical = content.lower().count('critical')
        n_major = content.lower().count('major')

        if response_file.exists():
            g.pass_(f'{len(reviews)} reviews, response file exists '
                    f'({n_critical} critical, {n_major} major mentions in synthesis)')
        else:
            if n_critical > 0:
                g.warn(f'{n_critical} critical items in synthesis, no response file')
            else:
                g.pass_(f'{len(reviews)} reviews found, no critical items')
    else:
        g.pass_(f'{len(reviews)} reviews found (no synthesis yet)')


def run_gates(skip_compile: bool = False, gate_filter: set[int] | None = None) -> int:
    """Run all quality gates.

    Args:
        skip_compile: Skip gate 4 (paper compilation).
        gate_filter: If set, only run these gate numbers.

    Returns:
        Exit code: 0 if all pass/warn, 1 if any fail.
    """
    gates = [
        (1, 'Tests', 'pytest suite', gate_tests),
        (2, 'Lint', 'ruff check', gate_lint),
        (3, 'Paper Numbers', 'verify claims vs source JSON', gate_paper_numbers),
        (4, 'Paper Compile', 'pdflatex x 3 + bibtex', gate_paper_compile),
        (5, 'Citations', 'verify bibliography', gate_citations),
        (6, 'Canonical JSON', 'check currency', gate_canonical_currency),
        (7, 'Registry', 'check completeness', gate_registry_completeness),
        (8, 'Review Coverage', 'check response status', gate_review_coverage),
    ]

    results = []
    for num, name, desc, func in gates:
        g = Gate(num, name, desc)

        if gate_filter and num not in gate_filter:
            g.status = 'SKIP'
            g.message = 'skipped by filter'
            results.append(g)
            continue

        if skip_compile and num == 4:
            g.status = 'SKIP'
            g.message = 'skipped by --skip-compile'
            results.append(g)
            continue

        try:
            func(g)
        except Exception as e:
            g.fail(f'Exception: {e}')

        results.append(g)

    # Print dashboard
    print(f"\n{'=' * 64}")
    print(f"  Pre-Submission Quality Gate — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 64}")

    for g in results:
        icon = {'PASS': 'PASS', 'WARN': 'WARN', 'FAIL': 'FAIL', 'SKIP': '----'}[g.status]
        color = {'PASS': '\033[32m', 'WARN': '\033[33m', 'FAIL': '\033[31m', 'SKIP': '\033[90m'}[g.status]
        reset = '\033[0m'
        print(f"  {color}{icon}{reset}  Gate {g.number}: {g.name:20s}  {g.message}")
        for d in g.details[:3]:
            print(f"         {d}")

    print(f"{'─' * 64}")

    n_pass = sum(1 for g in results if g.status == 'PASS')
    n_warn = sum(1 for g in results if g.status == 'WARN')
    n_fail = sum(1 for g in results if g.status == 'FAIL')
    n_skip = sum(1 for g in results if g.status == 'SKIP')
    total = len(results) - n_skip

    if n_fail == 0:
        verdict = '\033[32mREADY FOR SUBMISSION\033[0m' if n_warn == 0 else '\033[33mREVIEW WARNINGS\033[0m'
    else:
        verdict = '\033[31mNOT READY\033[0m'

    print(f"  {n_pass}/{total} passed, {n_warn} warnings, {n_fail} failed")
    print(f"  Verdict: {verdict}")
    print(f"{'=' * 64}\n")

    return 1 if n_fail > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Pre-submission quality gate (8 checks)')
    parser.add_argument('--skip-compile', action='store_true',
                        help='Skip paper compilation gate')
    parser.add_argument('--gates', type=str, default=None,
                        help='Comma-separated gate numbers to run (e.g., 1,2,3)')
    args = parser.parse_args()

    gate_filter = None
    if args.gates:
        gate_filter = {int(g.strip()) for g in args.gates.split(',')}

    exit_code = run_gates(
        skip_compile=args.skip_compile,
        gate_filter=gate_filter,
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
