"""
Extract review issues from synthesis markdown into issue_registry.yaml.

Parses CRITICAL/MAJOR/MINOR sections from synthesis files and updates
the persistent issue registry. New items get status=open; existing items
keep their current status.

Usage:
    python scripts/extract_review_issues.py
    python scripts/extract_review_issues.py --synthesis path/to/synthesis.md
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEWS_DIR = REPO_ROOT / "paper" / "review" / "reviews"
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
    """Save YAML file with readable formatting."""
    import yaml

    with open(path, "w") as f:
        f.write("# Paper Review Issue Registry\n")
        f.write("# Updated by: scripts/extract_review_issues.py\n")
        f.write(f"# Last updated: {date.today().isoformat()}\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=100)


def find_latest_synthesis() -> Path | None:
    """Find the most recent synthesis file."""
    if not REVIEWS_DIR.exists():
        return None
    syntheses = sorted(REVIEWS_DIR.glob("synthesis_*.md"))
    return syntheses[-1] if syntheses else None


def parse_table_row(line: str) -> dict | None:
    """Parse a markdown table row into issue fields.

    Handles formats like:
        | C1 | Spectral Entropy d=0.83->0.53 ... | Both | Fix all ... | **FIXING** |
        | M1 | Missing post-hoc ... | Statistician | Add Nemenyi ... |
    """
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 3:
        return None

    issue_id = cells[0].strip()
    if not re.match(r"^[CMm]\d+$", issue_id):
        return None

    return {
        "id": issue_id,
        "summary": cells[1].strip() if len(cells) > 1 else "",
        "reviewers": cells[2].strip() if len(cells) > 2 else "",
        "action": cells[3].strip() if len(cells) > 3 else "",
        "status_hint": cells[4].strip() if len(cells) > 4 else "",
    }


def parse_synthesis(path: Path) -> dict:
    """Parse a synthesis markdown file into structured issues.

    Returns:
        Dict with keys: critical, major, minor (lists of issue dicts),
        plus metadata.
    """
    content = path.read_text()
    lines = content.split("\n")

    result = {"critical": [], "major": [], "minor": [], "source": path.name}

    # Extract date from filename (synthesis_YYYYMMDD.md)
    date_match = re.search(r"synthesis_(\d{8})", path.name)
    if date_match:
        d = date_match.group(1)
        result["date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    current_section = None

    for line in lines:
        stripped = line.strip()

        # Detect section headers
        if re.match(r"^###?\s+CRITICAL", stripped, re.IGNORECASE):
            current_section = "critical"
            continue
        elif re.match(r"^###?\s+MAJOR", stripped, re.IGNORECASE):
            current_section = "major"
            continue
        elif re.match(r"^###?\s+MINOR", stripped, re.IGNORECASE):
            current_section = "minor"
            continue
        elif stripped.startswith("##") and current_section:
            current_section = None
            continue

        if current_section is None:
            continue

        # Parse table rows for CRITICAL/MAJOR
        if current_section in ("critical", "major") and "|" in stripped:
            parsed = parse_table_row(stripped)
            if parsed:
                result[current_section].append(parsed)

        # Parse list items for MINOR
        if current_section == "minor" and stripped.startswith("- "):
            text = stripped[2:].strip()
            if text and not text.startswith("---"):
                result["minor"].append({"summary": text})

    return result


def merge_into_registry(registry: dict, parsed: dict) -> tuple[int, int]:
    """Merge parsed synthesis issues into existing registry.

    New items get status=open. Existing items keep their current status.

    Returns:
        (new_count, updated_count)
    """
    issues = registry.setdefault("issues", [])
    existing_ids = {issue["id"]: i for i, issue in enumerate(issues)}

    new_count = 0
    updated_count = 0
    source = parsed.get("source", "unknown")
    review_date = parsed.get("date", date.today().isoformat())

    # Process CRITICAL and MAJOR (have IDs)
    for severity, items in [("CRITICAL", parsed["critical"]), ("MAJOR", parsed["major"])]:
        for item in items:
            issue_id = item["id"]
            if issue_id in existing_ids:
                # Update summary if changed, keep status
                idx = existing_ids[issue_id]
                if issues[idx].get("summary") != item["summary"]:
                    issues[idx]["summary"] = item["summary"]
                    updated_count += 1
            else:
                # New issue
                new_issue = {
                    "id": issue_id,
                    "severity": severity,
                    "summary": item["summary"],
                    "source": source,
                    "first_seen": review_date,
                    "status": "open",
                    "fix_description": "",
                    "paper_locations": [],
                    "tags": [],
                }
                issues.append(new_issue)
                new_count += 1

    # Process MINOR (no IDs — assign auto IDs)
    existing_minor_summaries = {
        issue["summary"] for issue in issues if issue.get("severity") == "MINOR"
    }
    minor_max = max(
        (int(issue["id"][1:]) for issue in issues
         if issue.get("id", "").startswith("m") and issue["id"][1:].isdigit()),
        default=0,
    )

    for item in parsed["minor"]:
        if item["summary"] not in existing_minor_summaries:
            minor_max += 1
            new_issue = {
                "id": f"m{minor_max}",
                "severity": "MINOR",
                "summary": item["summary"],
                "source": source,
                "first_seen": review_date,
                "status": "open",
            }
            issues.append(new_issue)
            new_count += 1

    # Update meta
    meta = registry.setdefault("meta", {})
    meta["last_review"] = review_date
    meta["review_count"] = meta.get("review_count", 0) + 1

    return new_count, updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract review issues from synthesis into registry"
    )
    parser.add_argument(
        "--synthesis", type=Path, default=None,
        help="Path to synthesis markdown (default: latest in paper/review/reviews/)",
    )
    parser.add_argument(
        "--registry", type=Path, default=REGISTRY_PATH,
        help=f"Path to issue registry YAML (default: {REGISTRY_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and report without writing registry",
    )
    args = parser.parse_args()

    # Find synthesis
    synthesis_path = args.synthesis or find_latest_synthesis()
    if not synthesis_path or not synthesis_path.exists():
        print("ERROR: No synthesis file found", file=sys.stderr)
        sys.exit(1)
    print(f"Parsing: {synthesis_path.name}")

    # Parse synthesis
    parsed = parse_synthesis(synthesis_path)
    print(f"  Found: {len(parsed['critical'])} CRITICAL, "
          f"{len(parsed['major'])} MAJOR, {len(parsed['minor'])} MINOR")

    # Load or initialize registry
    if args.registry.exists():
        registry = load_yaml(args.registry)
    else:
        registry = {"meta": {"paper": "qcml_geometric_sde", "created": date.today().isoformat()}}

    # Merge
    new_count, updated_count = merge_into_registry(registry, parsed)
    print(f"  Registry: {new_count} new, {updated_count} updated")

    if args.dry_run:
        print("  (dry-run, not writing)")
        return

    # Save
    save_yaml(registry, args.registry)
    print(f"  Written: {args.registry.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
