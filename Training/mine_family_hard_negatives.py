#!/usr/bin/env python3
"""
Mine intra-family hard negatives from the KBOB material taxonomy.

For each training query, finds materials in the same taxonomic family as
the query's positives, but which are NOT themselves positives for that query.
These teach the model to discriminate between semantically close materials
within the same family (e.g. Balkenschichtholz vs Konstruktionsvollholz).

Produces JSONL in the same format as mine_hard_negatives.py, so both
outputs can be concatenated and fed to prepare_training_data.py via
``--hard-negatives-file``.
"""

import argparse
import json
from pathlib import Path

from text_normalization import normalize_material_key


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY = (
    SCRIPT_DIR.parent
    / "Training"
    / "query_generation"
    / "sources"
    / "material_ökobilanz.txt"
)


# ---------------------------------------------------------------------------
# Taxonomy loader
# ---------------------------------------------------------------------------

def load_material_families(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Parse ``material_ökobilanz.txt`` with ``#Family`` headers.

    Returns
    -------
    mat_to_family : dict mapping material name → family key
    family_to_mats : dict mapping family key → set of material names
    """
    mat_to_family: dict[str, str] = {}
    family_to_mats: dict[str, set[str]] = {}
    current_family = "unknown"

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            current_family = stripped.lstrip("#").strip()
            continue
        mat_to_family[stripped] = current_family
        family_to_mats.setdefault(current_family, set()).add(stripped)

    return mat_to_family, family_to_mats


# ---------------------------------------------------------------------------
# Core mining
# ---------------------------------------------------------------------------

def mine_family_hard_negatives(
    query_lines: list[str],
    mapping_lines: list[str],
    mat_to_family: dict[str, str],
    family_to_mats: dict[str, set[str]],
    max_negatives_per_query: int,
    cross_family_extras: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Return one record per unique query that has intra-family negatives.

    Parameters
    ----------
    cross_family_extras:
        Optional mapping of positive-material → list of extra hard-negative
        materials from *other* families that should always be included for
        any query whose positive set contains the key.  These are appended
        after the intra-family candidates and are subject to the same
        ``max_negatives_per_query`` cap.

        Example::

            {"Gussasphalt": ["Dichtungsbahn bituminös"]}

        This ensures that queries like ``IfcPavement FLEXIBLE Bitumenmischgut``
        (positive: Gussasphalt) learn to distinguish Gussasphalt from
        Dichtungsbahn bituminös even though they live in different taxonomy
        families and would never be mined by the intra-family pass alone.
    """

    # Aggregate each unique query's positive set
    by_query: dict[str, set[str]] = {}
    query_order: list[str] = []
    for q, m in zip(query_lines, mapping_lines):
        q = q.strip()
        if not q:
            continue
        positives = {p.strip() for p in m.split("|") if p.strip()}
        if q not in by_query:
            by_query[q] = set()
            query_order.append(q)
        by_query[q].update(positives)

    records: list[dict] = []
    total_negatives = 0

    for query in query_order:
        positives = by_query[query]

        # Collect which families the positives span
        positive_families: set[str] = set()
        for p in positives:
            fam = mat_to_family.get(p)
            if fam:
                positive_families.add(fam)

        if not positive_families:
            continue

        # Gather all same-family materials that are NOT own positives
        positive_keys = {normalize_material_key(p) for p in positives}
        candidates: list[str] = []
        for fam in sorted(positive_families):
            for mat in sorted(family_to_mats.get(fam, [])):
                if normalize_material_key(mat) not in positive_keys:
                    candidates.append(mat)

        # Append cross-family extras (e.g. Dichtungsbahn bituminös for Gussasphalt)
        if cross_family_extras:
            for positive in positives:
                for extra in cross_family_extras.get(positive, []):
                    if normalize_material_key(extra) not in positive_keys and extra not in candidates:
                        candidates.append(extra)

        if not candidates:
            continue

        hard_negatives = candidates[:max_negatives_per_query]
        total_negatives += len(hard_negatives)

        records.append({
            "query": query,
            "positives": sorted(positives),
            "hard_negatives": hard_negatives,
            "source_details_file": "intra_family_mining",
        })

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine intra-family hard negatives from KBOB material taxonomy."
    )
    p.add_argument(
        "--query-file",
        required=True,
        help="Path to generated_queries_without_exposure.txt (one query per line).",
    )
    p.add_argument(
        "--mapping-file",
        required=True,
        help="Path to mapping_generated_queries_without_exposure.txt (pipe-separated positives per line).",
    )
    p.add_argument(
        "--material-taxonomy",
        default=str(DEFAULT_TAXONOMY),
        help="Path to material_ökobilanz.txt with #Family headers (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output JSONL path.",
    )
    p.add_argument(
        "--max-negatives-per-query",
        type=int,
        default=5,
        help="Maximum number of intra-family hard negatives per query (default: 5).",
    )
    p.add_argument(
        "--cross-family-extras",
        default=None,
        help=(
            'JSON string mapping positive material → extra HN list from other families. '
            'Example: \'{"Gussasphalt": ["Dichtungsbahn bituminös"]}\''
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    query_file = Path(args.query_file).expanduser().resolve()
    mapping_file = Path(args.mapping_file).expanduser().resolve()
    taxonomy_file = Path(args.material_taxonomy).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()

    if not query_file.is_file():
        raise FileNotFoundError(f"Query file not found: {query_file}")
    if not mapping_file.is_file():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    if not taxonomy_file.is_file():
        raise FileNotFoundError(f"Material taxonomy file not found: {taxonomy_file}")

    query_lines = query_file.read_text(encoding="utf-8").splitlines()
    mapping_lines = mapping_file.read_text(encoding="utf-8").splitlines()

    if len(query_lines) != len(mapping_lines):
        raise ValueError(
            f"Line count mismatch: {len(query_lines)} queries vs {len(mapping_lines)} mappings."
        )

    mat_to_family, family_to_mats = load_material_families(taxonomy_file)

    print(f"Taxonomy: {len(mat_to_family)} materials in {len(family_to_mats)} families")
    for fam in sorted(family_to_mats):
        print(f"  {fam}: {len(family_to_mats[fam])} materials")

    cross_family_extras: dict[str, list[str]] | None = None
    if args.cross_family_extras:
        cross_family_extras = json.loads(args.cross_family_extras)
        print(f"Cross-family extras: {cross_family_extras}")

    records = mine_family_hard_negatives(
        query_lines=query_lines,
        mapping_lines=mapping_lines,
        mat_to_family=mat_to_family,
        family_to_mats=family_to_mats,
        max_negatives_per_query=args.max_negatives_per_query,
        cross_family_extras=cross_family_extras,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_hn = sum(len(r["hard_negatives"]) for r in records)
    print(f"Queries with family hard negatives: {len(records)}")
    print(f"Total hard negatives: {total_hn}")
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
