"""
Citation verification via Semantic Scholar API.

Parses the thebibliography entries from the paper's .tex file, queries
Semantic Scholar for each entry, and verifies authors/year/journal match.
Flags suspicious entries.

Usage:
    python paper/verify_citations.py
    python paper/verify_citations.py --tex paper/qcml_geometric_sde.tex
    python paper/verify_citations.py --delay 3.5
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = REPO_ROOT / "paper" / "qcml_geometric_sde.tex"

# Semantic Scholar rate limit: 100 requests per 5 minutes = 1 per 3 seconds
DEFAULT_DELAY = 3.1  # seconds between API calls


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BibEntry:
    """A parsed bibliography entry from \\bibitem."""

    key: str
    raw_text: str
    title: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    entry_type: str = "unknown"  # article, book, preprint, techreport


@dataclass
class VerificationResult:
    """Result of verifying one citation."""

    key: str
    status: str  # VERIFIED, PARTIAL, NOT_FOUND, SUSPICIOUS, ERROR, SKIPPED
    detail: str = ""
    ss_title: str = ""
    ss_year: Optional[int] = None
    ss_authors: list = field(default_factory=list)
    title_match: bool = False
    year_match: bool = False


# ---------------------------------------------------------------------------
# Parsing thebibliography entries
# ---------------------------------------------------------------------------
def parse_thebibliography(tex_path: Path) -> list:
    """Parse \\bibitem entries from a LaTeX file.

    Returns a list of BibEntry objects extracted from the
    \\begin{thebibliography} ... \\end{thebibliography} block.
    """
    with open(tex_path) as f:
        content = f.read()

    # Extract the thebibliography block
    bib_match = re.search(
        r"\\begin\{thebibliography\}.*?\n(.*?)\\end\{thebibliography\}",
        content,
        re.DOTALL,
    )
    if not bib_match:
        print(f"WARNING: No \\begin{{thebibliography}} found in {tex_path}")
        return []

    bib_block = bib_match.group(1)

    # Split on \bibitem
    entries = re.split(r"\\bibitem\{", bib_block)
    entries = [e.strip() for e in entries if e.strip()]

    results = []
    for entry_text in entries:
        # Extract key (everything before the closing brace)
        key_match = re.match(r"([^}]+)\}\s*(.*)", entry_text, re.DOTALL)
        if not key_match:
            continue

        key = key_match.group(1).strip()
        body = key_match.group(2).strip()

        entry = BibEntry(key=key, raw_text=body)
        _parse_entry_body(entry, body)
        results.append(entry)

    return results


def _parse_entry_body(entry: BibEntry, body: str):
    """Extract title, authors, year, journal from a bibitem body."""
    # Clean LaTeX commands for parsing
    clean = body
    clean = re.sub(r"\\newblock\s*", " ", clean)
    clean = re.sub(r"\\doi\{[^}]*\}", "", clean)
    clean = re.sub(r"\\href\{[^}]*\}\{[^}]*\}", "", clean)
    clean = re.sub(r"\\emph\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"\\texttt\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"\\textbf\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"\\textrm\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"\\\\\s*", " ", clean)
    clean = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", clean)
    clean = re.sub(r"~", " ", clean)
    clean = re.sub(r"\s+", " ", clean)

    # Extract year (4-digit number, typically 19xx or 20xx)
    year_matches = re.findall(r"\b((?:19|20)\d{2})\b", clean)
    if year_matches:
        entry.year = int(year_matches[-1])  # Usually the last year is the publication year

    # Extract title (text between ``...'' or "...")
    title_match = re.search(r"``([^']+)''", body)
    if not title_match:
        title_match = re.search(r'"([^"]+)"', body)
    if title_match:
        entry.title = _clean_latex(title_match.group(1))
        entry.entry_type = "article"
    else:
        # Book: title in \emph{...}
        emph_match = re.search(r"\\emph\{([^}]+)\}", body)
        if emph_match:
            entry.title = _clean_latex(emph_match.group(1))
            entry.entry_type = "book"

    # Detect preprints
    if "arxiv" in body.lower() or "preprint" in body.lower():
        entry.entry_type = "preprint"

    # Detect tech reports
    if "white paper" in body.lower() or "technical" in body.lower():
        entry.entry_type = "techreport"

    # Extract journal from \emph{...} for articles
    if entry.entry_type == "article":
        journal_matches = re.findall(r"\\emph\{([^}]+)\}", body)
        if journal_matches:
            entry.journal = _clean_latex(journal_matches[0])

    # Extract authors (text before the first `` or \emph)
    # Authors are typically listed before the title
    author_text = body
    # Cut at the title
    for delimiter in ["``", r"\emph{"]:
        idx = author_text.find(delimiter)
        if idx > 0:
            author_text = author_text[:idx]
            break

    # Parse author names
    author_text = _clean_latex(author_text)
    author_text = author_text.strip().rstrip(",").strip()
    if author_text:
        # Split on "and" or ","
        parts = re.split(r"\band\b|,", author_text)
        entry.authors = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


def _clean_latex(text: str) -> str:
    """Remove common LaTeX markup from text."""
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\\\s", " ", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Semantic Scholar API
# ---------------------------------------------------------------------------
def query_semantic_scholar(title: str, year: Optional[int] = None) -> Optional[dict]:
    """Query Semantic Scholar API for a paper by title.

    Returns the best matching paper dict or None.
    """
    import urllib.request
    import json as json_mod

    # Build query
    query = title
    if len(query) > 200:
        query = query[:200]

    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={quote_plus(query)}"
        f"&limit=5"
        f"&fields=title,year,authors,venue,externalIds"
    )
    if year:
        url += f"&year={year - 1}-{year + 1}"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "QCML-Paper-Citation-Verifier/1.0")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json_mod.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

    papers = data.get("data", [])
    if not papers:
        return None

    # Find best match by title similarity
    best = None
    best_score = 0
    for paper in papers:
        score = _title_similarity(title, paper.get("title", ""))
        if score > best_score:
            best_score = score
            best = paper

    if best and best_score > 0.4:
        best["_match_score"] = best_score
        return best

    # Return first result if no good match
    papers[0]["_match_score"] = _title_similarity(title, papers[0].get("title", ""))
    return papers[0]


def _title_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two titles (word-level)."""
    if not a or not b:
        return 0.0
    words_a = set(re.sub(r"[^\w\s]", "", a.lower()).split())
    words_b = set(re.sub(r"[^\w\s]", "", b.lower()).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------
def verify_entry(entry: BibEntry, delay: float) -> VerificationResult:
    """Verify a single bibliography entry against Semantic Scholar.

    Parameters
    ----------
    entry : BibEntry
        The parsed bibliography entry.
    delay : float
        Seconds to wait before making the API call (rate limiting).

    Returns
    -------
    VerificationResult
    """
    result = VerificationResult(key=entry.key)

    # Skip tech reports and entries without titles
    if entry.entry_type == "techreport":
        result.status = "SKIPPED"
        result.detail = "Technical report / white paper (not indexed by Semantic Scholar)"
        return result

    if not entry.title:
        result.status = "SKIPPED"
        result.detail = "Could not extract title from entry"
        return result

    # Rate limit
    time.sleep(delay)

    # Query API
    ss_result = query_semantic_scholar(entry.title, entry.year)

    if ss_result is None:
        result.status = "NOT_FOUND"
        result.detail = f"No results for: '{entry.title}'"
        return result

    if "error" in ss_result:
        result.status = "ERROR"
        result.detail = f"API error: {ss_result['error']}"
        return result

    # Extract Semantic Scholar data
    ss_title = ss_result.get("title", "")
    ss_year = ss_result.get("year")
    ss_authors_raw = ss_result.get("authors", [])
    ss_authors = [a.get("name", "") for a in ss_authors_raw]
    match_score = ss_result.get("_match_score", 0)

    result.ss_title = ss_title
    result.ss_year = ss_year
    result.ss_authors = ss_authors

    # Title match
    result.title_match = match_score > 0.6

    # Year match
    result.year_match = (
        entry.year is not None
        and ss_year is not None
        and abs(entry.year - ss_year) <= 1
    )

    # Determine status
    if result.title_match and result.year_match:
        result.status = "VERIFIED"
        result.detail = (
            f"Title match={match_score:.2f}, year: {entry.year}=={ss_year}"
        )
    elif result.title_match:
        result.status = "PARTIAL"
        year_detail = f"paper={entry.year}, SS={ss_year}" if ss_year else "year not found"
        result.detail = f"Title match={match_score:.2f}, year mismatch ({year_detail})"
    elif match_score > 0.3:
        result.status = "SUSPICIOUS"
        result.detail = (
            f"Weak title match={match_score:.2f}. "
            f"SS title: '{ss_title}'"
        )
    else:
        result.status = "NOT_FOUND"
        result.detail = (
            f"No good match (best score={match_score:.2f}). "
            f"SS title: '{ss_title}'"
        )

    return result


# ---------------------------------------------------------------------------
# BIB file parsing (if a .bib file exists instead)
# ---------------------------------------------------------------------------
def _find_bib_from_tex(tex_path: Path) -> Optional[Path]:
    """Extract the .bib file path from a \\bibliography{...} command in a .tex file."""
    with open(tex_path) as f:
        content = f.read()
    match = re.search(r"\\bibliography\{([^}]+)\}", content)
    if match:
        bib_name = match.group(1).strip()
        if not bib_name.endswith(".bib"):
            bib_name += ".bib"
        return tex_path.parent / bib_name
    return None


def parse_bib_file(bib_path: Path) -> list:
    """Parse a .bib file into BibEntry objects.

    Uses bibtexparser if available, otherwise regex fallback.
    """
    try:
        import bibtexparser

        with open(bib_path) as f:
            bib_db = bibtexparser.load(f)

        entries = []
        for item in bib_db.entries:
            entry = BibEntry(
                key=item.get("ID", "unknown"),
                raw_text=str(item),
                title=item.get("title", "").strip("{}"),
                year=int(item["year"]) if "year" in item else None,
                journal=item.get("journal", item.get("booktitle", "")),
            )
            author_str = item.get("author", "")
            entry.authors = [a.strip() for a in author_str.split(" and ")]

            if "arxiv" in str(item).lower():
                entry.entry_type = "preprint"
            elif entry.journal:
                entry.entry_type = "article"
            else:
                entry.entry_type = "unknown"

            entries.append(entry)
        return entries

    except ImportError:
        # Regex fallback for .bib files
        return _parse_bib_regex(bib_path)


def _parse_bib_regex(bib_path: Path) -> list:
    """Regex-based .bib parser as fallback."""
    with open(bib_path) as f:
        content = f.read()

    entries = []
    # Match @type{key, ... }
    pattern = r"@(\w+)\{([^,]+),\s*(.*?)\n\}"
    for match in re.finditer(pattern, content, re.DOTALL):
        entry_type = match.group(1).lower()
        key = match.group(2).strip()
        body = match.group(3)

        entry = BibEntry(key=key, raw_text=body)

        # Extract fields
        for field_match in re.finditer(r"(\w+)\s*=\s*\{([^}]*)\}", body):
            field_name = field_match.group(1).lower()
            field_value = field_match.group(2).strip()

            if field_name == "title":
                entry.title = field_value
            elif field_name == "year":
                try:
                    entry.year = int(field_value)
                except ValueError:
                    pass
            elif field_name in ("journal", "booktitle"):
                entry.journal = field_value
            elif field_name == "author":
                entry.authors = [a.strip() for a in field_value.split(" and ")]

        entry.entry_type = entry_type
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Verify citations against Semantic Scholar API"
    )
    parser.add_argument(
        "--tex",
        type=str,
        default=str(DEFAULT_TEX),
        help="Path to the LaTeX file with thebibliography (default: paper/qcml_geometric_sde.tex)",
    )
    parser.add_argument(
        "--bib",
        type=str,
        default=None,
        help="Path to a .bib file (if used instead of thebibliography)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between API calls (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse citations but do not query Semantic Scholar",
    )
    args = parser.parse_args()

    # Parse citations
    # Strategy: --bib flag takes priority, then auto-detect .bib from \bibliography{},
    # then fall back to \bibitem parsing from the .tex file.
    if args.bib:
        bib_path = Path(args.bib)
        if not bib_path.exists():
            print(f"ERROR: {bib_path} not found")
            sys.exit(1)
        entries = parse_bib_file(bib_path)
        print(f"Parsed {len(entries)} entries from {bib_path.name}")
    else:
        tex_path = Path(args.tex)
        if not tex_path.exists():
            print(f"ERROR: {tex_path} not found")
            sys.exit(1)

        # Auto-detect .bib file referenced via \bibliography{...}
        bib_path = _find_bib_from_tex(tex_path)
        if bib_path and bib_path.exists():
            entries = parse_bib_file(bib_path)
            print(f"Parsed {len(entries)} entries from {bib_path.name} (auto-detected)")
        else:
            entries = parse_thebibliography(tex_path)
            print(f"Parsed {len(entries)} entries from {tex_path.name}")

    if not entries:
        print("No citations found.")
        sys.exit(0)

    # Print parsed entries summary
    print("\nParsed citations:")
    for e in entries:
        year_str = str(e.year) if e.year else "????"
        title_short = e.title[:60] + "..." if len(e.title) > 60 else e.title
        print(f"  [{e.key}] ({year_str}) {title_short}")

    if args.dry_run:
        print("\n--dry-run specified; skipping Semantic Scholar queries.")
        return

    # Verify each entry
    print(f"\nQuerying Semantic Scholar ({args.delay}s between requests)...")
    print("=" * 72)

    results = []
    for i, entry in enumerate(entries):
        print(f"  [{i + 1}/{len(entries)}] {entry.key}...", end=" ", flush=True)
        result = verify_entry(entry, delay=args.delay if i > 0 else 0)
        results.append(result)
        print(result.status)

    # Print report
    print("\n" + "=" * 72)
    print("CITATION VERIFICATION REPORT")
    print("=" * 72)

    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    # Group by status
    for status in ["VERIFIED", "PARTIAL", "SUSPICIOUS", "NOT_FOUND", "ERROR", "SKIPPED"]:
        group = [r for r in results if r.status == status]
        if not group:
            continue

        print(f"\n--- {status} ({len(group)}) ---")
        for r in group:
            print(f"  [{r.key}] {r.detail}")
            if r.ss_title and status in ("SUSPICIOUS", "PARTIAL"):
                print(f"    SS title: {r.ss_title}")
                if r.ss_authors:
                    authors_str = ", ".join(r.ss_authors[:3])
                    if len(r.ss_authors) > 3:
                        authors_str += f" + {len(r.ss_authors) - 3} more"
                    print(f"    SS authors: {authors_str}")

    # Summary
    print("\n" + "=" * 72)
    total = len(results)
    verified = status_counts.get("VERIFIED", 0)
    partial = status_counts.get("PARTIAL", 0)
    suspicious = status_counts.get("SUSPICIOUS", 0)
    not_found = status_counts.get("NOT_FOUND", 0)
    errors = status_counts.get("ERROR", 0)
    skipped = status_counts.get("SKIPPED", 0)

    print(f"Total: {total} citations")
    print(f"  Verified:   {verified}")
    print(f"  Partial:    {partial}")
    print(f"  Suspicious: {suspicious}")
    print(f"  Not found:  {not_found}")
    print(f"  Errors:     {errors}")
    print(f"  Skipped:    {skipped}")

    coverage = (verified + partial) / max(total - skipped, 1) * 100
    print(f"\nVerification coverage: {coverage:.0f}% ({verified + partial}/{total - skipped})")

    if suspicious > 0:
        print(f"\nWARNING: {suspicious} citation(s) flagged as suspicious -- review manually.")

    if not_found > 0:
        print(f"NOTE: {not_found} citation(s) not found on Semantic Scholar.")
        print("  This may be normal for very recent preprints or niche publications.")


if __name__ == "__main__":
    main()
