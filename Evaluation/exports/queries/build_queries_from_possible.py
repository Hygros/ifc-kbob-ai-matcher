from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


FILTERED_TOKENS = {"", "NONE", "NOTDEFINED"}
DEFAULT_VALUE = ""
CONCRETE_CASTING_METHODS = ("INSITU", "PRECAST")
IFCPILE_CASTING_EXCLUSIONS = {
    "BORED": {"PRECAST"},
    "DRIVEN": {"INSITU"},
    "JETGROUTING": {"PRECAST"},
}

# Stahlbeton erbt Festigkeits-/Expositionsklassen von Beton
MATERIAL_ALIAS = {"Stahlbeton": "Beton"}


def is_filtered_token(value: str) -> bool:
    return value.strip().upper() in FILTERED_TOKENS


def unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def iter_clean_lines(path: Path):
    text = path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        yield line_number, line


def normalize_exposure_bundle(raw_exposure: str) -> str:
    tokens = [token.strip() for token in raw_exposure.replace(",", " ").split()]
    tokens = [token for token in tokens if token and not is_filtered_token(token)]
    return " ".join(tokens)


def parse_entity_predefined(path: Path):
    result = defaultdict(list)
    warnings = []

    for line_number, line in iter_clean_lines(path):
        if ":" not in line:
            warnings.append(f"{path.name}:{line_number} -> Missing ':' delimiter, line skipped")
            continue

        entity, right = line.split(":", 1)
        entity = entity.strip()
        if not entity:
            warnings.append(f"{path.name}:{line_number} -> Empty entity key, line skipped")
            continue

        predefined_types = [token.strip() for token in right.split() if token.strip()]
        predefined_types = [token for token in predefined_types if not is_filtered_token(token)]

        if not predefined_types:
            predefined_types = [""]

        result[entity].extend(predefined_types)

    for entity, predefined_types in result.items():
        result[entity] = unique_preserve_order(predefined_types)

    return dict(result), warnings


def parse_entity_material(path: Path):
    """Parse (entity, predefined_type) -> materials mapping.

    Line formats:
      Entity PredefinedType: Material1 Material2   -> key (Entity, PredefinedType)
      Entity: Material1 Material2                  -> key (Entity, "") as default
    """
    result: dict[tuple[str, str], list[str]] = defaultdict(list)
    warnings = []

    for line_number, line in iter_clean_lines(path):
        if ":" not in line:
            warnings.append(f"{path.name}:{line_number} -> Missing ':' delimiter, line skipped")
            continue

        left, right = line.split(":", 1)
        left = left.strip()
        if not left:
            warnings.append(f"{path.name}:{line_number} -> Empty key, line skipped")
            continue

        parts = left.split(None, 1)
        entity = parts[0]
        predefined_type = parts[1].strip() if len(parts) > 1 else ""

        materials = [token.strip() for token in right.split() if token.strip()]
        materials = [m for m in materials if not is_filtered_token(m)]

        key = (entity, predefined_type)
        result[key].extend(materials)

    for key in result:
        result[key] = unique_preserve_order(result[key])

    return dict(result), warnings


def parse_entity_diameters(path: Path):
    """Parse entity -> diameters mapping."""
    result = defaultdict(list)
    warnings = []

    for line_number, line in iter_clean_lines(path):
        if ":" not in line:
            warnings.append(f"{path.name}:{line_number} -> Missing ':' delimiter, line skipped")
            continue

        entity, right = line.split(":", 1)
        entity = entity.strip()
        if not entity:
            warnings.append(f"{path.name}:{line_number} -> Empty entity key, line skipped")
            continue

        diameters = [token.strip() for token in right.split() if token.strip()]
        diameters = [d for d in diameters if not is_filtered_token(d)]

        result[entity].extend(diameters)

    for entity, diameters in result.items():
        result[entity] = unique_preserve_order(diameters)

    return dict(result), warnings


def parse_material_strength_exposure(path: Path):
    """Parse the combined material -> [(strength, exposure)] file.

    Supported formats:
      Beton:  XC4 XF1: C25/30, C30/37          <- exposure_bundle: strength, strength
              XC4 XD1 XF2: C25/30, C30/37       <- indented continuation line
      Stahl: S235; S355; S460                    <- semicolon-separated strengths (no exposure)
      Betonstahl: B500A; B500B; B500C; B700B
    """
    result = defaultdict(list)
    warnings = []

    text = path.read_text(encoding="utf-8")
    current_material = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        is_continuation = raw_line[0] in (" ", "\t") if raw_line else False

        if not is_continuation and ":" in stripped:
            # New material line
            material, rest = stripped.split(":", 1)
            material = material.strip()
            if not material or is_filtered_token(material):
                warnings.append(f"{path.name}:{line_number} -> Invalid material key, line skipped")
                current_material = None
                continue
            current_material = material
            rest = rest.strip()
            if rest:
                _parse_material_rest(
                    result, current_material, rest, path.name, line_number, warnings
                )

        elif is_continuation and current_material:
            _parse_material_rest(
                result, current_material, stripped, path.name, line_number, warnings
            )

        else:
            warnings.append(f"{path.name}:{line_number} -> Could not parse line, skipped")

    for material, pairs in result.items():
        result[material] = unique_preserve_order(pairs)

    return dict(result), warnings


def _parse_material_rest(
    result: dict, material: str, rest: str, filename: str, line_number: int, warnings: list
):
    """Parse the right-hand side of a material line or a continuation line.

    Two sub-formats:
      1) "ExposureBundle: Strength1, Strength2"  (contains ':')
      2) "S235; S355; S460"                      (semicolon-separated, no exposure)
    """
    if ":" in rest:
        exposure_raw, strengths_raw = rest.split(":", 1)
        exposure_bundle = normalize_exposure_bundle(exposure_raw)
        strengths = [s.strip() for s in strengths_raw.split(",") if s.strip()]
        strengths = [s for s in strengths if not is_filtered_token(s)]
        if not strengths:
            warnings.append(
                f"{filename}:{line_number} -> No valid strengths after exposure '{exposure_bundle}'"
            )
            return
        for strength in strengths:
            result[material].append((strength, exposure_bundle))
    else:
        entries = [e.strip() for e in rest.split(";") if e.strip()]
        for entry in entries:
            tokens = entry.split()
            for token in tokens:
                token = token.strip().rstrip(",")
                if token and not is_filtered_token(token):
                    result[material].append((token, DEFAULT_VALUE))


def diameter_sort_key(value: str):
    if value == "":
        return (1, 10**9)
    try:
        return (0, int(value))
    except ValueError:
        return (0, value)


def _lookup_strength_exposure(
    material: str,
    material_strength_exposure: dict[str, list[tuple[str, str]]],
) -> list[tuple[str, str]]:
    """Lookup (strength, exposure) pairs for a material, resolving aliases."""
    pairs = material_strength_exposure.get(material)
    if pairs is not None:
        return pairs
    alias = MATERIAL_ALIAS.get(material)
    if alias:
        return material_strength_exposure.get(alias, [])
    return []


def _lookup_materials(
    entity: str,
    predefined_type: str,
    entity_material: dict[tuple[str, str], list[str]],
) -> list[str]:
    """Lookup materials for (entity, predefined_type) with fallback to (entity, '')."""
    materials = entity_material.get((entity, predefined_type))
    if materials is not None:
        return materials
    return entity_material.get((entity, ""), [])


def _is_concrete_family(material: str) -> bool:
    """Return True for Beton and aliases that map to Beton."""
    resolved = MATERIAL_ALIAS.get(material, material)
    return resolved.casefold() == "beton"


def _filter_casting_methods(
    entity: str,
    predefined_type: str,
    casting_methods: tuple[str, ...],
) -> tuple[str, ...]:
    """Apply IFCPILE-specific casting restrictions for selected predefined types."""
    if entity.casefold() != "ifcpile":
        return casting_methods

    blocked_methods = IFCPILE_CASTING_EXCLUSIONS.get(predefined_type.strip().upper())
    if not blocked_methods:
        return casting_methods

    return tuple(
        method for method in casting_methods if not method or method.upper() not in blocked_methods
    )


def generate_queries(
    entity_predefined: dict[str, list[str]],
    entity_material: dict[tuple[str, str], list[str]],
    material_strength_exposure: dict[str, list[tuple[str, str]]],
    entity_diameters: dict[str, list[str]],
    include_exposure: bool = True,
):
    raw_rows = []
    skipped_without_material = 0
    materials_without_strength = 0

    for entity, predefined_types in entity_predefined.items():
        predefined_values = predefined_types if predefined_types else [""]
        entity_diam = entity_diameters.get(entity, [""])
        if not entity_diam:
            entity_diam = [""]

        for predefined_type in predefined_values:
            materials = _lookup_materials(entity, predefined_type, entity_material)
            if not materials:
                skipped_without_material += 1
                continue

            for material in materials:
                pairs = _lookup_strength_exposure(material, material_strength_exposure)

                if not pairs:
                    pairs = [(DEFAULT_VALUE, DEFAULT_VALUE)]
                    materials_without_strength += 1

                # Durchmesser + CastingMethod nur fuer Beton / Stahlbeton
                is_concrete_family = _is_concrete_family(material)
                diameters = entity_diam if is_concrete_family else [DEFAULT_VALUE]
                casting_methods = (
                    CONCRETE_CASTING_METHODS if is_concrete_family else (DEFAULT_VALUE,)
                )
                casting_methods = _filter_casting_methods(entity, predefined_type, casting_methods)
                if not casting_methods:
                    continue

                for strength, exposure_bundle in pairs:
                    effective_exposure = exposure_bundle if include_exposure else DEFAULT_VALUE
                    for diameter in diameters:
                        for casting_method in casting_methods:
                            row = (
                                entity,
                                predefined_type,
                                material,
                                strength,
                                effective_exposure,
                                diameter,
                                casting_method,
                            )
                            raw_rows.append(row)

    unique_rows = sorted(
        set(raw_rows),
        key=lambda row: (
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            diameter_sort_key(row[5]),
            row[6],
        ),
    )

    stats = {
        "raw_rows": len(raw_rows),
        "unique_rows": len(unique_rows),
        "skipped_without_material": skipped_without_material,
        "materials_without_strength": materials_without_strength,
    }
    return unique_rows, stats


def write_queries(path: Path, rows: list[tuple[str, str, str, str, str, str, str]]):
    lines = [" ".join(part for part in row if part) for row in rows]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SBERT-style query lines from possible*.txt files"
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--entity-predefined-file",
        type=Path,
        default=Path("possible_entities-predefinedtypes.txt"),
    )
    parser.add_argument(
        "--entity-material-file",
        type=Path,
        default=Path("possible_entity-material.txt"),
    )
    parser.add_argument(
        "--entity-diameter-file",
        type=Path,
        default=Path("possible_entity-durchmesser.txt"),
    )
    parser.add_argument(
        "--material-strength-exposure-file",
        type=Path,
        default=Path("possible_material-strength_exposure.txt"),
    )
    parser.add_argument("--output-file", type=Path, default=Path("generated_queries.txt"))
    parser.add_argument(
        "--output-file-without-exposure",
        type=Path,
        default=Path("generated_queries_without_exposure.txt"),
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def main():
    args = parse_args()

    base_dir = args.base_dir.resolve()
    entity_predefined_path = resolve_path(base_dir, args.entity_predefined_file)
    entity_material_path = resolve_path(base_dir, args.entity_material_file)
    entity_diameter_path = resolve_path(base_dir, args.entity_diameter_file)
    material_strength_exposure_path = resolve_path(base_dir, args.material_strength_exposure_file)
    output_with_exposure_path = resolve_path(base_dir, args.output_file)
    output_without_exposure_path = resolve_path(base_dir, args.output_file_without_exposure)

    required_files = [
        entity_predefined_path,
        entity_material_path,
        entity_diameter_path,
        material_strength_exposure_path,
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        missing_lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required input file(s):\n"
            f"{missing_lines}\n"
            "Create the missing possible_*.txt files in the queries folder."
        )

    entity_predefined, warn_ep = parse_entity_predefined(entity_predefined_path)
    entity_material, warn_em = parse_entity_material(entity_material_path)
    entity_diameter, warn_ed = parse_entity_diameters(entity_diameter_path)
    material_strength_exposure, warn_mse = parse_material_strength_exposure(
        material_strength_exposure_path
    )

    all_warnings = warn_ep + warn_em + warn_ed + warn_mse

    rows_with_exposure, stats_with_exposure = generate_queries(
        entity_predefined=entity_predefined,
        entity_material=entity_material,
        material_strength_exposure=material_strength_exposure,
        entity_diameters=entity_diameter,
        include_exposure=True,
    )
    rows_without_exposure, stats_without_exposure = generate_queries(
        entity_predefined=entity_predefined,
        entity_material=entity_material,
        material_strength_exposure=material_strength_exposure,
        entity_diameters=entity_diameter,
        include_exposure=False,
    )

    output_with_exposure_path.parent.mkdir(parents=True, exist_ok=True)
    output_without_exposure_path.parent.mkdir(parents=True, exist_ok=True)
    write_queries(output_with_exposure_path, rows_with_exposure)
    write_queries(output_without_exposure_path, rows_without_exposure)

    print(
        "[OK] Wrote "
        f"{stats_with_exposure['unique_rows']} unique queries WITH exposure to "
        f"{output_with_exposure_path}"
    )
    print(
        "[OK] Wrote "
        f"{stats_without_exposure['unique_rows']} unique queries WITHOUT exposure to "
        f"{output_without_exposure_path}"
    )
    print(f"[INFO] Raw rows before dedup (with exposure): {stats_with_exposure['raw_rows']}")
    print(f"[INFO] Raw rows before dedup (without exposure): {stats_without_exposure['raw_rows']}")
    print(
        "[INFO] Skipped (entity, predefined_type) without material mapping: "
        f"{stats_with_exposure['skipped_without_material']}"
    )
    print(
        "[INFO] Material combinations without strength/exposure (only entity+material): "
        f"{stats_with_exposure['materials_without_strength']}"
    )
    if all_warnings:
        print(f"[WARN] {len(all_warnings)} parse warning(s):")
        for warning in all_warnings[:20]:
            print(f"  - {warning}")
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings) - 20} more")


if __name__ == "__main__":
    main()