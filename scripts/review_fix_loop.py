"""
Review-Fix-Verify loop orchestrator.

Chains the feedback loop: review -> extract issues -> track fixes -> verify -> re-review.

Usage:
    python scripts/review_fix_loop.py status       # Show issue status dashboard
    python scripts/review_fix_loop.py extract      # Extract issues from latest synthesis
    python scripts/review_fix_loop.py verify       # Verify fixes (run make verify + check registry)
    python scripts/review_fix_loop.py re-review    # Targeted re-review of unfixed critical items
"""

import argparse
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "paper" / "review" / "issue_registry.yaml"


def load_yaml(path: Path) -> dict:
    """Load YAML file."""
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)


def save_yaml(data: dict, path: Path) -> None:
    """Save YAML file."""
    import yaml

    with open(path, "w") as f:
        f.write("# Paper Review Issue Registry\n")
        f.write("# Updated by: scripts/review_fix_loop.py\n")
        f.write(f"# Last updated: {date.today().isoformat()}\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=100)


# ── Status ────────────────────────────────────────────────────────────

def cmd_status(args):
    """Show issue status dashboard."""
    if not REGISTRY_PATH.exists():
        print("No issue registry found. Run: make review-extract")
        return 1

    registry = load_yaml(REGISTRY_PATH)
    issues = registry.get("issues", [])
    meta = registry.get("meta", {})

    if not issues:
        print("No issues in registry.")
        return 0

    # Group by severity
    by_severity = {"CRITICAL": [], "MAJOR": [], "MINOR": []}
    for issue in issues:
        sev = issue.get("severity", "MINOR").upper()
        by_severity.setdefault(sev, []).append(issue)

    # Status counts
    all_statuses = Counter(issue.get("status", "open") for issue in issues)

    # Header
    print(f"\n{'=' * 70}")
    print(f"  Review Issue Tracker — Last review: {meta.get('last_review', 'unknown')}")
    print(f"{'=' * 70}")

    # Summary bar
    total = len(issues)
    verified = sum(1 for i in issues if i.get("status") == "verified")
    fixed = sum(1 for i in issues if i.get("status") == "fixed")
    fixing = sum(1 for i in issues if i.get("status") == "fixing")
    wontfix = sum(1 for i in issues if i.get("status") in ("wontfix", "deferred"))
    remaining = total - verified - fixed - wontfix

    print(f"\n  Total: {total}  |  "
          f"\033[32mVerified: {verified}\033[0m  |  "
          f"\033[36mFixed: {fixed}\033[0m  |  "
          f"\033[33mFixing: {fixing}\033[0m  |  "
          f"\033[90mWontfix: {wontfix}\033[0m  |  "
          f"\033[31mOpen: {remaining - fixing}\033[0m")

    # Detail by severity
    for sev in ["CRITICAL", "MAJOR", "MINOR"]:
        items = by_severity.get(sev, [])
        if not items:
            continue

        color = {"CRITICAL": "\033[31m", "MAJOR": "\033[33m", "MINOR": "\033[90m"}[sev]
        reset = "\033[0m"

        print(f"\n  {color}{sev}{reset} ({len(items)})")
        print(f"  {'─' * 66}")

        for issue in items:
            status = issue.get("status", "open")
            status_icon = {
                "open": "\033[31m  OPEN\033[0m",
                "fixing": "\033[33mFIXING\033[0m",
                "fixed": "\033[36m FIXED\033[0m",
                "verified": "\033[32mVERIFY\033[0m",
                "wontfix": "\033[90mWONTFX\033[0m",
                "deferred": "\033[90mDEFER \033[0m",
            }.get(status, status)

            issue_id = issue.get("id", "?")
            summary = issue.get("summary", "")
            # Truncate summary for display
            max_len = 50
            if len(summary) > max_len:
                summary = summary[:max_len - 3] + "..."

            print(f"    {status_icon}  {issue_id:4s}  {summary}")

    # Submission readiness
    critical_open = sum(
        1 for i in by_severity.get("CRITICAL", [])
        if i.get("status") not in ("verified", "wontfix")
    )
    major_open = sum(
        1 for i in by_severity.get("MAJOR", [])
        if i.get("status") not in ("verified", "wontfix", "deferred")
    )

    print(f"\n{'─' * 70}")
    if critical_open == 0 and major_open == 0:
        print("  \033[32mSUBMITTABLE\033[0m — all CRITICAL/MAJOR issues resolved")
    elif critical_open == 0:
        print(f"  \033[33mALMOST READY\033[0m — {major_open} MAJOR issue(s) remaining")
    else:
        print(f"  \033[31mNOT READY\033[0m — {critical_open} CRITICAL, {major_open} MAJOR remaining")
    print(f"{'=' * 70}\n")

    return 0


# ── Extract ───────────────────────────────────────────────────────────

def cmd_extract(args):
    """Extract issues from latest synthesis."""
    script = REPO_ROOT / "scripts" / "extract_review_issues.py"
    if not script.exists():
        print("ERROR: extract_review_issues.py not found", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(script)]
    if args.dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


# ── Verify ────────────────────────────────────────────────────────────

def cmd_verify(args):
    """Verify fixes: run make verify, then check registry items."""
    print("Phase 1: Running make verify...")
    verify_result = subprocess.run(
        ["make", "verify"], cwd=REPO_ROOT, capture_output=True, text=True,
    )

    if verify_result.returncode != 0:
        print(f"  FAIL: make verify failed")
        print(verify_result.stdout[-500:] if verify_result.stdout else "")
        return 1
    print("  PASS: Paper numbers match canonical JSON")

    if not REGISTRY_PATH.exists():
        print("\nNo issue registry found. Run: make review-extract")
        return 0

    registry = load_yaml(REGISTRY_PATH)
    issues = registry.get("issues", [])
    promoted = 0

    print("\nPhase 2: Checking fixed issues...")
    for issue in issues:
        if issue.get("status") != "fixed":
            continue

        issue_id = issue.get("id", "?")
        summary = issue.get("summary", "")[:50]

        # For data-error tags, make verify already confirmed numbers match
        tags = issue.get("tags", [])
        if "data-error" in tags:
            print(f"  {issue_id}: data error verified by make verify -> VERIFIED")
            issue["status"] = "verified"
            issue["verification"] = {
                "method": "make_verify",
                "date": date.today().isoformat(),
                "passed": True,
            }
            promoted += 1
        else:
            # For non-data issues, we can only confirm they were marked fixed
            print(f"  {issue_id}: {summary} — marked fixed (manual verification needed)")

    if promoted > 0:
        save_yaml(registry, REGISTRY_PATH)
        print(f"\n  Promoted {promoted} issue(s) to VERIFIED")

    return 0


# ── Re-review ─────────────────────────────────────────────────────────

def cmd_rereview(args):
    """Targeted re-review of unfixed critical items."""
    if not REGISTRY_PATH.exists():
        print("No issue registry. Run: make review-extract")
        return 1

    registry = load_yaml(REGISTRY_PATH)
    issues = registry.get("issues", [])

    # Find unresolved CRITICAL items
    unresolved = [
        i for i in issues
        if i.get("severity") == "CRITICAL"
        and i.get("status") not in ("verified", "wontfix")
    ]

    if not unresolved:
        print("All CRITICAL issues resolved. No re-review needed.")
        return 0

    print(f"Found {len(unresolved)} unresolved CRITICAL issue(s):")
    for issue in unresolved:
        print(f"  {issue['id']}: {issue.get('summary', '')[:60]}")

    if args.dry_run:
        print("\n(dry-run, skipping review)")
        return 0

    print(f"\nRunning targeted re-review (quick mode)...")
    result = subprocess.run(
        ["make", "review", "ARGS=--quick"], cwd=REPO_ROOT,
    )

    if result.returncode == 0:
        print("\nRe-review complete. Run 'make review-extract' to update registry.")

    return result.returncode


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Review-Fix-Verify loop orchestrator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status
    sub = subparsers.add_parser("status", help="Show issue status dashboard")
    sub.set_defaults(func=cmd_status)

    # extract
    sub = subparsers.add_parser("extract", help="Extract issues from latest synthesis")
    sub.add_argument("--dry-run", action="store_true")
    sub.set_defaults(func=cmd_extract)

    # verify
    sub = subparsers.add_parser("verify", help="Verify fixes against registry")
    sub.set_defaults(func=cmd_verify)

    # re-review
    sub = subparsers.add_parser("re-review", help="Targeted re-review of unfixed criticals")
    sub.add_argument("--dry-run", action="store_true")
    sub.set_defaults(func=cmd_rereview)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
