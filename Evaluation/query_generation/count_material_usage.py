from __future__ import annotations

import argparse
from pathlib import Path


def load_canonical_materials(materials_file: Path) -> list[str]:
    materials: list[str] = []
    for raw_line in materials_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        materials.append(line)
    return materials


def count_materials(
    mapping_files: list[Path],
    canonical_materials: list[str],
) -> tuple[dict[str, int], set[str]]:
    counts = {material: 0 for material in canonical_materials}
    canonical_set = set(canonical_materials)
    unknown_materials: set[str] = set()

    for mapping_file in mapping_files:
        for raw_line in mapping_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Mapping lines may contain multiple canonical options separated by "|".
            for token in line.split("|"):
                material = token.strip()
                if not material:
                    continue
                if material in canonical_set:
                    counts[material] += 1
                else:
                    unknown_materials.add(material)

    return counts, unknown_materials


def write_summary(output_file: Path, canonical_materials: list[str], counts: dict[str, int]) -> None:
    lines = ["material\tcount"]
    lines.extend(f"{material}\t{counts[material]}" for material in canonical_materials)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Count canonical material usage in a mapping*.txt file."
    )
    parser.add_argument(
        "--materials-file",
        type=Path,
        default=script_dir / "material_ökobilanz.txt",
        help="Path to canonical material list (default: material_ökobilanz.txt).",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional single mapping file. If omitted, --mapping-glob is used.",
    )
    parser.add_argument(
        "--mapping-glob",
        type=str,
        default="mapping*.txt",
        help="Glob pattern (relative to script directory) used when --mapping-file is omitted.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=script_dir / "material_ökobilanz_usage_summary_without_exposure.txt",
        help="Output summary txt path.",
    )
    parser.add_argument(
        "--strict-unknown",
        action="store_true",
        help="Fail if materials appear in mapping file that are missing from materials-file.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    canonical_materials = load_canonical_materials(args.materials_file)

    if args.mapping_file is not None:
        mapping_files = [args.mapping_file]
    else:
        script_dir = Path(__file__).resolve().parent
        mapping_files = sorted(script_dir.glob(args.mapping_glob))

    if not mapping_files:
        print("No mapping files found.")
        return 1

    counts, unknown_materials = count_materials(mapping_files, canonical_materials)

    if unknown_materials:
        print(
            "Warning: Unknown materials found (not in canonical list): "
            + ", ".join(sorted(unknown_materials))
        )
        if args.strict_unknown:
            return 1

    write_summary(args.output_file, canonical_materials, counts)
    print("Processed mapping files:")
    for mapping_file in mapping_files:
        print(f"- {mapping_file}")
    print(f"Wrote summary: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
