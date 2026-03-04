"""
Verify paper claims against source experiment data.

Reads memory/results_registry.yaml which maps every number in the paper
to its source JSON file and JSON path. Checks each claim.

Usage:
    python scripts/verify_paper_numbers.py
    python scripts/verify_paper_numbers.py --registry path/to/registry.yaml
    python scripts/verify_paper_numbers.py --tolerance 0.02
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "memory" / "results_registry.yaml"


# ---------------------------------------------------------------------------
# YAML loading (with fallback)
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> dict:
    """Load YAML file, falling back to simple parser if PyYAML unavailable."""
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # Minimal YAML-subset parser for the registry format
        return _parse_simple_yaml(path)


def _parse_simple_yaml(path: Path) -> dict:
    """Parse the subset of YAML used by the results registry.

    Handles a flat list of claims under a 'claims' key, where each claim
    has scalar fields (strings, numbers) at one indentation level.
    """
    with open(path) as f:
        lines = f.readlines()

    result = {}
    current_key = None
    current_list = None
    current_item = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.lstrip()

        # Skip blanks and comments
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(stripped)

        # Top-level key (e.g., "claims:" or "tolerance:")
        if indent == 0 and ":" in stripped:
            if current_item is not None and current_list is not None:
                current_list.append(current_item)
                current_item = None

            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            if val == "" or val is None:
                current_key = key
                current_list = []
                result[key] = current_list
            else:
                result[key] = _cast_value(val)
                current_key = None
                current_list = None
            continue

        # List item start (e.g., "  - name: ...")
        if stripped.startswith("- "):
            if current_item is not None and current_list is not None:
                current_list.append(current_item)
            current_item = {}
            # Parse the key-value after "- "
            item_content = stripped[2:]
            if ":" in item_content:
                k, _, v = item_content.partition(":")
                current_item[k.strip()] = _cast_value(v.strip())
            continue

        # Continuation of a list item (e.g., "    source_json: ...")
        if current_item is not None and ":" in stripped:
            k, _, v = stripped.partition(":")
            current_item[k.strip()] = _cast_value(v.strip())
            continue

    # Flush last item
    if current_item is not None and current_list is not None:
        current_list.append(current_item)

    return result


def _cast_value(val: str):
    """Cast a YAML scalar value to Python type."""
    if val == "" or val == "null" or val == "~":
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    # Strip quotes
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
    # Try numeric
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


# ---------------------------------------------------------------------------
# JSON path traversal
# ---------------------------------------------------------------------------
def resolve_json_path(data: dict, path: str):
    """Resolve a dot-notation path through a JSON object.

    Supports dictionary keys and integer list indices.
    Example paths:
        "summary.median_d.Berry Phase Rate"
        "results.Berry Phase Rate.2008_gfc.d"
        "results.0.score"

    If jsonpath-ng is available, uses it for full JSONPath support.
    Otherwise falls back to simple dot-notation traversal.
    """
    try:
        from jsonpath_ng import parse as jp_parse

        # Convert dot notation to JSONPath bracket notation for keys with spaces
        jp_expr = "$"
        for part in _split_path(path):
            if part.isdigit():
                jp_expr += f"[{part}]"
            else:
                jp_expr += f".'{part}'" if " " in part else f".{part}"

        expr = jp_parse(jp_expr)
        matches = [m.value for m in expr.find(data)]
        if matches:
            return matches[0]
        # Fall through to simple traversal
    except ImportError:
        pass

    # Simple dot-notation traversal
    current = data
    for part in _split_path(path):
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                raise KeyError(f"Key '{part}' not found. Available: {list(current.keys())[:10]}")
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as e:
                raise KeyError(f"Invalid list index '{part}': {e}")
        else:
            raise KeyError(f"Cannot traverse into {type(current).__name__} with key '{part}'")

    return current


def _split_path(path: str) -> list:
    """Split a dot-notation path, respecting keys that contain dots
    when they are quoted or contain spaces.

    For simplicity, this splits on '.' but reassembles parts that look
    like they belong to a single key (e.g., multi-word method names).
    Uses a heuristic: if a part matches a known method name fragment,
    merge with neighbors.

    In practice the registry should use '/' or bracket notation for
    ambiguous keys. This implementation uses '.' as separator.
    """
    return path.split(".")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_claim(claim: dict, tolerance: float) -> dict:
    """Verify a single claim against its source data.

    Returns a result dict with keys: name, status, detail.
    """
    name = claim.get("name", "unnamed")
    source_json = claim.get("source_json")
    json_path = claim.get("json_path")
    paper_value = claim.get("paper_value")

    if not source_json or not json_path:
        return {
            "name": name,
            "status": "SKIP",
            "detail": "Missing source_json or json_path",
        }

    # Resolve relative paths from repo root
    json_file = REPO_ROOT / source_json
    if not json_file.exists():
        return {
            "name": name,
            "status": "FAIL",
            "detail": f"Source file not found: {source_json}",
        }

    try:
        with open(json_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "name": name,
            "status": "FAIL",
            "detail": f"Cannot load JSON: {e}",
        }

    try:
        actual_value = resolve_json_path(data, json_path)
    except KeyError as e:
        return {
            "name": name,
            "status": "FAIL",
            "detail": f"JSON path error: {e}",
        }

    # Compare
    if paper_value is None:
        return {
            "name": name,
            "status": "INFO",
            "detail": f"Source value = {actual_value} (no paper_value to compare)",
        }

    try:
        paper_num = float(paper_value)
        actual_num = float(actual_value)
    except (TypeError, ValueError):
        # String comparison
        if str(paper_value).strip() == str(actual_value).strip():
            return {"name": name, "status": "PASS", "detail": f"Exact match: {actual_value}"}
        else:
            return {
                "name": name,
                "status": "FAIL",
                "detail": f"Mismatch: paper='{paper_value}', source='{actual_value}'",
            }

    diff = abs(paper_num - actual_num)
    if diff <= tolerance:
        return {
            "name": name,
            "status": "PASS",
            "detail": f"paper={paper_num:.4f}, source={actual_num:.4f}, diff={diff:.4f}",
        }
    else:
        return {
            "name": name,
            "status": "FAIL",
            "detail": f"paper={paper_num:.4f}, source={actual_num:.4f}, diff={diff:.4f} > tol={tolerance}",
        }


# ---------------------------------------------------------------------------
# Sample registry template
# ---------------------------------------------------------------------------
SAMPLE_REGISTRY = """\
# Results Registry: maps every claimed number in the paper to source data.
#
# Each claim has:
#   name:         Human-readable description of the claim
#   section:      Paper section where this number appears
#   paper_value:  The value as written in the paper
#   source_json:  Path to source JSON (relative to repo root)
#   json_path:    Dot-notation path to the value in the JSON
#
# Canonical run: 2026-02-28, 13 methods x 14 crises, yfinance data
# Run: python scripts/verify_paper_numbers.py
#
# Tolerance for numeric comparisons (default 0.02 for Cohen's d values).
tolerance: 0.02

claims:
  - name: "Berry Phase Rate median d"
    section: "Table 4"
    paper_value: 0.63
    source_json: "experiments/outputs/regime_detection/causal_comparison_20260228_112418.json"
    json_path: "summary.median_d.Berry Phase Rate"

  - name: "Random Forest median d"
    section: "Table 4"
    paper_value: 0.37
    source_json: "experiments/outputs/regime_detection/causal_comparison_20260228_112418.json"
    json_path: "summary.median_d.Random Forest"

  - name: "Friedman chi-squared"
    section: "Table 4 caption"
    paper_value: 43.3
    source_json: "experiments/outputs/regime_detection/causal_comparison_20260228_112418.json"
    json_path: "summary.friedman_chi_sq"
"""


def create_sample_registry(path: Path):
    """Write a sample registry YAML if none exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_REGISTRY)
    print(f"Created sample registry at: {path}")
    print("Edit it to add your paper's claimed numbers, then re-run.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Verify paper claims against source experiment data"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=str(DEFAULT_REGISTRY),
        help=f"Path to results_registry.yaml (default: {DEFAULT_REGISTRY.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Numeric tolerance for comparisons (overrides registry value; default 0.01)",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample registry YAML and exit",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)

    if args.create_sample:
        create_sample_registry(registry_path)
        return

    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        print("Creating sample registry...")
        create_sample_registry(registry_path)
        return

    # Load registry
    registry = load_yaml(registry_path)
    claims = registry.get("claims", [])
    if not claims:
        print("No claims found in registry. Add entries under 'claims:' key.")
        sys.exit(1)

    # Determine tolerance
    tolerance = args.tolerance
    if tolerance is None:
        tolerance = registry.get("tolerance", 0.01)
        if tolerance is None:
            tolerance = 0.01

    print(f"Verifying {len(claims)} claim(s) with tolerance={tolerance}")
    print(f"Registry: {registry_path}")
    print("=" * 72)

    # Verify each claim
    results = []
    for claim in claims:
        result = verify_claim(claim, tolerance)
        results.append(result)

    # Print report
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_info = sum(1 for r in results if r["status"] == "INFO")

    for r in results:
        status = r["status"]
        if status == "PASS":
            marker = "PASS"
        elif status == "FAIL":
            marker = "FAIL"
        elif status == "SKIP":
            marker = "SKIP"
        else:
            marker = "INFO"

        print(f"  [{marker}] {r['name']}")
        print(f"         {r['detail']}")

    print("=" * 72)
    print(f"Results: {n_pass} passed, {n_fail} failed, {n_skip} skipped, {n_info} info")

    if n_fail > 0:
        print("\nFAILED claims need attention -- paper numbers may be stale.")
        sys.exit(1)
    else:
        print("\nAll verified claims match source data.")
        sys.exit(0)


if __name__ == "__main__":
    main()
