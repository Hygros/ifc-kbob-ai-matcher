from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


QueryRow = tuple[str, str, str, str, str, str, str]


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

# Strength classes exclusive to lean concrete (Magerbeton); never combine with PRECAST or Stahlbeton
LEAN_STRENGTHS = {"C12/15", "C16/20"}

# Only these entities may receive LEAN_STRENGTHS; all others are excluded
LEAN_STRENGTH_ENTITIES = {
    "IfcBuildingElementProxy",
    "IfcBuiltElement",
    "IfcCaissonFoundation",
    "IfcCourse",
    "IfcFooting",
    "IfcSlab",
    "IfcWall",
}


DEFAULT_POLICY = {
    "variant_modules": {
        "npk": {
            "enabled": True,
            "normal_grades": ["D", "E", "F", "G"],
            "deep_foundation_grades": ["H", "I", "K", "L"],
            "deep_foundation_targets": [
                "IfcPile::BORED",
                "IfcWall::RETAININGWALL",
            ],
            "entity_predefined_grades": {},
            "select_one_grade_per_query": True,
            "selection_seed": "npk-default",
            "emit_simplified_variant": True,
        },
        "steel_grade_alias": {
            "enabled": True,
            "strength_aliases": {
                "S235": ["S235JR", "S235J0"],
                "S355": ["S355JR", "S355J0"],
            },
        },
        "ptfe_alias": {
            "enabled": True,
            "entity_predefined_targets": ["IfcBearing::GUIDE"],
            "aliases": ["PTFE", "Teflon", "Polytetrafluoroethylene"],
            "emit_without_strength": True,
        },
        "pavement": {
            "enabled": True,
            "predefined_targets": ["PAVEMENT", "PAVING", "SIDEWALK", "WEARING", "FLEXIBLE"],
            "material_aliases": {
                "Asphalt": ["Asphaltbelag"],
                "Bitumen": ["Bitumenbelag"],
            },
            "emit_without_strength": True,
        },
        "aggregate": {
            "enabled": True,
            "predefined_targets": ["ARMOUR", "BALLASTBED", "FILTER", "PROTECTION", "SUPPORT"],
            "material_aliases": {
                "Kies": ["Aggregate"],
                "Naturstein": ["Aggregate"],
            },
            "emit_without_strength": True,
        },
    },
    "confusion_expansions": {
        "enabled": True,
        "drop_strength_for_materials": ["Beton", "Stahlbeton", "Stahl", "Holz", "Buche", "Eiche", "Fichte", "Tanne", "Lärche"],
        "drop_casting_for_concrete": True,
        "drop_strength_and_casting_for_concrete": True,
    },
    "balancing": {
        "max_variants_per_entity_predefined_material": 0,
    },
    "debug_exports": {
        "enabled": True,
    },
}


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


def _compose_query_text(row: QueryRow) -> str:
    parts = list(row)
    # Material field (index 2): replace underscores with spaces so that
    # multi-word tokens like Hochfester_Stahl appear as "Hochfester Stahl"
    # in the generated query text.  Predefined-type underscores (index 1,
    # e.g. COVER_PLATE) are intentionally kept.
    if parts[2]:
        parts[2] = parts[2].replace("_", " ")
    return " ".join(part for part in parts if part)


def _technical_dedupe_key_from_row(row: QueryRow) -> str:
    # Technical dedupe key only: trim + collapse whitespace. No semantic normalization.
    return " ".join(_compose_query_text(row).strip().split())


def extract_semantic_slots(record: QueryRow) -> dict[str, str]:
    return {
        "entity": record[0],
        "predefined_type": record[1],
        "material": record[2],
        "strength": record[3],
        "exposure": record[4],
        "diameter": record[5],
        "casting_method": record[6],
    }


def _build_row_from_slots(slots: dict[str, str], **overrides: str) -> QueryRow:
    resolved = {
        "entity": slots["entity"],
        "predefined_type": slots["predefined_type"],
        "material": slots["material"],
        "strength": slots["strength"],
        "exposure": slots["exposure"],
        "diameter": slots["diameter"],
        "casting_method": slots["casting_method"],
    }
    resolved.update(overrides)
    return (
        resolved["entity"],
        resolved["predefined_type"],
        resolved["material"],
        resolved["strength"],
        resolved["exposure"],
        resolved["diameter"],
        resolved["casting_method"],
    )


def _module_policy(policy: dict | None, module_name: str) -> dict:
    if not policy:
        return {}
    return policy.get("variant_modules", {}).get(module_name, {})


def _module_enabled(policy: dict | None, module_name: str) -> bool:
    return bool(_module_policy(policy, module_name).get("enabled", False))


def _entity_predefined_key(slots: dict[str, str]) -> str:
    return f"{slots['entity']}::{slots['predefined_type']}"


def _is_steel_material(material: str) -> bool:
    return material.strip().casefold() == "stahl"


def _deterministic_choice(options: list[str], selection_key: str) -> str:
    if len(options) == 1:
        return options[0]
    digest = hashlib.sha256(selection_key.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(options)
    return options[index]


def _apply_casting_alias(slots: dict[str, str], policy: dict | None) -> None:
    """Replace IFC casting token with German alias per deterministic seed."""
    if not policy or not slots["casting_method"]:
        return
    casting_cfg = policy.get("variant_modules", {}).get("casting_alias", {})
    if not casting_cfg.get("enabled", False):
        return
    aliases = casting_cfg.get("aliases", {}).get(slots["casting_method"])
    if not aliases or len(aliases) <= 1:
        return
    selection_seed = str(casting_cfg.get("selection_seed", "casting-default"))
    selection_key = "|".join([
        selection_seed,
        slots["entity"],
        slots["predefined_type"],
        slots["material"],
        slots["strength"],
        slots["diameter"],
        slots["casting_method"],
    ])
    slots["casting_method"] = _deterministic_choice(aliases, selection_key)


def _resolve_npk_grades(slots: dict[str, str], npk_policy: dict) -> list[str]:
    key = _entity_predefined_key(slots)

    explicit_map = npk_policy.get("entity_predefined_grades", {})
    explicit_grades = explicit_map.get(key)
    if explicit_grades is not None:
        return unique_preserve_order(
            [str(grade).strip() for grade in explicit_grades if str(grade).strip()]
        )

    deep_targets = {
        str(value).strip()
        for value in npk_policy.get("deep_foundation_targets", [])
        if str(value).strip()
    }
    if key in deep_targets:
        return unique_preserve_order(
            [
                str(grade).strip()
                for grade in npk_policy.get("deep_foundation_grades", [])
                if str(grade).strip()
            ]
        )

    if _is_concrete_family(slots["material"]):
        return unique_preserve_order(
            [
                str(grade).strip()
                for grade in npk_policy.get("normal_grades", [])
                if str(grade).strip()
            ]
        )

    return []


def generate_canonical_queries(slots: dict[str, str]) -> list[QueryRow]:
    return [
        (
            slots["entity"],
            slots["predefined_type"],
            slots["material"],
            slots["strength"],
            slots["exposure"],
            slots["diameter"],
            slots["casting_method"],
        )
    ]


def generate_variant_queries(slots: dict[str, str], policy: dict | None = None) -> list[QueryRow]:
    variants: list[QueryRow] = []
    module_config = policy.get("variant_modules", {}) if policy else {}

    # NPK module
    if _module_enabled(policy, "npk") and _is_concrete_family(slots["material"]) and slots["strength"] not in LEAN_STRENGTHS:
        npk_policy = module_config.get("npk", {})
        grades = _resolve_npk_grades(slots, npk_policy)
        if grades and bool(npk_policy.get("select_one_grade_per_query", True)):
            selection_seed = str(npk_policy.get("selection_seed", "npk-default"))
            selection_key = "|".join(
                [
                    selection_seed,
                    slots["entity"],
                    slots["predefined_type"],
                    slots["material"],
                    slots["strength"],
                    slots["diameter"],
                    slots["casting_method"],
                ]
            )
            grades = [_deterministic_choice(grades, selection_key)]

        for grade in grades:
            npk_material = f"{slots['material']} NPK {grade}".strip()
            variants.append(_build_row_from_slots(slots, material=npk_material))
            if npk_policy.get("emit_simplified_variant", True):
                variants.append(
                    _build_row_from_slots(
                        slots,
                        material=npk_material,
                        strength=DEFAULT_VALUE,
                        exposure=DEFAULT_VALUE,
                        casting_method=DEFAULT_VALUE,
                    )
                )

    # Steel grade aliases (S235 -> S235JR)
    if _module_enabled(policy, "steel_grade_alias") and _is_steel_material(slots["material"]):
        alias_policy = module_config.get("steel_grade_alias", {})
        alias_strengths = alias_policy.get("strength_aliases", {}).get(slots["strength"], [])
        for alias_strength in unique_preserve_order(
            [str(value).strip() for value in alias_strengths if str(value).strip()]
        ):
            variants.append(_build_row_from_slots(slots, strength=alias_strength))

    # PTFE/Teflon/Polytetrafluoroethylene aliases for configured entity/predefined pairs
    if _module_enabled(policy, "ptfe_alias"):
        ptfe_policy = module_config.get("ptfe_alias", {})
        targets = {str(value).strip() for value in ptfe_policy.get("entity_predefined_targets", [])}
        if _entity_predefined_key(slots) in targets:
            aliases = [str(value).strip() for value in ptfe_policy.get("aliases", []) if str(value).strip()]
            for alias in unique_preserve_order(aliases):
                if ptfe_policy.get("emit_without_strength", True):
                    variants.append(
                        _build_row_from_slots(
                            slots,
                            material=alias,
                            strength=DEFAULT_VALUE,
                            exposure=DEFAULT_VALUE,
                            diameter=DEFAULT_VALUE,
                            casting_method=DEFAULT_VALUE,
                        )
                    )
                else:
                    variants.append(_build_row_from_slots(slots, material=alias))

    # Pavement module
    if _module_enabled(policy, "pavement"):
        pavement_policy = module_config.get("pavement", {})
        targets = {str(value).strip().upper() for value in pavement_policy.get("predefined_targets", [])}
        if slots["predefined_type"].strip().upper() in targets:
            aliases = pavement_policy.get("material_aliases", {}).get(slots["material"], [])
            aliases = [str(value).strip() for value in aliases if str(value).strip()]
            for alias in unique_preserve_order(aliases):
                if pavement_policy.get("emit_without_strength", True):
                    variants.append(
                        _build_row_from_slots(
                            slots,
                            material=alias,
                            strength=DEFAULT_VALUE,
                            exposure=DEFAULT_VALUE,
                            casting_method=DEFAULT_VALUE,
                        )
                    )
                else:
                    variants.append(_build_row_from_slots(slots, material=alias))

    # Aggregate module
    if _module_enabled(policy, "aggregate"):
        aggregate_policy = module_config.get("aggregate", {})
        targets = {str(value).strip().upper() for value in aggregate_policy.get("predefined_targets", [])}
        if slots["predefined_type"].strip().upper() in targets:
            aliases = aggregate_policy.get("material_aliases", {}).get(slots["material"], [])
            aliases = [str(value).strip() for value in aliases if str(value).strip()]
            for alias in unique_preserve_order(aliases):
                if aggregate_policy.get("emit_without_strength", True):
                    variants.append(
                        _build_row_from_slots(
                            slots,
                            material=alias,
                            strength=DEFAULT_VALUE,
                            exposure=DEFAULT_VALUE,
                            casting_method=DEFAULT_VALUE,
                        )
                    )
                else:
                    variants.append(_build_row_from_slots(slots, material=alias))

    return variants


def generate_confusion_targeted_queries(
    slots: dict[str, str],
    confusion_profile: dict | None,
) -> list[QueryRow]:
    if not confusion_profile or not confusion_profile.get("enabled", False):
        return []

    variants: list[QueryRow] = []
    drop_strength_materials = {
        str(value).strip().casefold()
        for value in confusion_profile.get("drop_strength_for_materials", [])
        if str(value).strip()
    }

    if (
        slots["strength"]
        and slots["material"].strip().casefold() in drop_strength_materials
    ):
        variants.append(
            _build_row_from_slots(
                slots,
                strength=DEFAULT_VALUE,
                exposure=DEFAULT_VALUE,
            )
        )

    if (
        confusion_profile.get("drop_casting_for_concrete", False)
        and slots["casting_method"]
        and _is_concrete_family(slots["material"])
    ):
        variants.append(
            _build_row_from_slots(
                slots,
                casting_method=DEFAULT_VALUE,
            )
        )

    # Combined: drop strength AND casting simultaneously for concrete-family
    # materials. This generates bare queries like "IfcBeam Beton" that appear
    # frequently in real IFC exports without any strength or casting markers.
    if (
        confusion_profile.get("drop_strength_and_casting_for_concrete", False)
        and slots["strength"]
        and slots["casting_method"]
        and slots["material"].strip().casefold() in drop_strength_materials
        and _is_concrete_family(slots["material"])
    ):
        variants.append(
            _build_row_from_slots(
                slots,
                strength=DEFAULT_VALUE,
                exposure=DEFAULT_VALUE,
                casting_method=DEFAULT_VALUE,
            )
        )

    # Drop material token for steel-grade queries.  This generates queries like
    # "IfcBeam LINTEL S235JR" (without the explicit "Stahl" material keyword)
    # so the model learns to recognise steel grades as implicit steel indicators.
    if (
        confusion_profile.get("drop_material_for_steel_grade", False)
        and _is_steel_material(slots["material"])
        and slots["strength"]
    ):
        steel_grade_tokens = {
            str(t).strip().upper()
            for t in confusion_profile.get("steel_grade_tokens", [])
            if str(t).strip()
        }
        if slots["strength"].strip().upper() in steel_grade_tokens:
            variants.append(
                _build_row_from_slots(
                    slots,
                    material=DEFAULT_VALUE,
                )
            )

    # Bare entity+predefined queries (no material, no strength, no casting).
    # Generates queries like "IfcWall RETAININGWALL" so the model learns to
    # handle elements where only the entity type and predefined type are known.
    bare_targets = {
        str(t).strip()
        for t in confusion_profile.get("bare_entity_predefined_queries", [])
        if str(t).strip()
    }
    if bare_targets and slots["material"]:
        ep_key = f"{slots['entity']} {slots['predefined_type']}".strip()
        if ep_key in bare_targets:
            variants.append(
                _build_row_from_slots(
                    slots,
                    material=DEFAULT_VALUE,
                    strength=DEFAULT_VALUE,
                    exposure=DEFAULT_VALUE,
                    diameter=DEFAULT_VALUE,
                    casting_method=DEFAULT_VALUE,
                )
            )

    return variants


def deduplicate_and_balance(queries: list[QueryRow], policy: dict | None = None) -> list[QueryRow]:
    balancing_policy = (policy or {}).get("balancing", {})
    max_per_group = int(balancing_policy.get("max_variants_per_entity_predefined_material", 0) or 0)

    seen_keys: set[str] = set()
    deduplicated: list[QueryRow] = []

    for query in queries:
        dedupe_key = _technical_dedupe_key_from_row(query)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduplicated.append(query)

    sorted_rows = sorted(
        deduplicated,
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

    if max_per_group <= 0:
        return sorted_rows

    group_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    balanced_rows: list[QueryRow] = []
    for row in sorted_rows:
        group_key = (row[0], row[1], row[2])
        if group_counts[group_key] >= max_per_group:
            continue
        group_counts[group_key] += 1
        balanced_rows.append(row)
    return balanced_rows


def export_query_artifacts(path: Path, rows: list[QueryRow]) -> None:
    lines = [_compose_query_text(row) for row in rows]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def export_debug_artifacts(path: Path, rows: list[QueryRow]) -> None:
    debug_dir = path.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = debug_dir / f"{path.stem}_debug.jsonl"
    csv_path = debug_dir / f"{path.stem}_debug.csv"

    jsonl_lines = []
    for index, row in enumerate(rows, start=1):
        payload = {
            "index": index,
            "entity": row[0],
            "predefined_type": row[1],
            "material": row[2],
            "strength": row[3],
            "exposure": row[4],
            "diameter": row[5],
            "casting_method": row[6],
            "query": _compose_query_text(row),
            "dedupe_key": _technical_dedupe_key_from_row(row),
        }
        jsonl_lines.append(json.dumps(payload, ensure_ascii=False))

    jsonl_content = "\n".join(jsonl_lines)
    if jsonl_lines:
        jsonl_content += "\n"
    jsonl_path.write_text(jsonl_content, encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "entity",
                "predefined_type",
                "material",
                "strength",
                "exposure",
                "diameter",
                "casting_method",
                "query",
                "dedupe_key",
            ]
        )
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    index,
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    _compose_query_text(row),
                    _technical_dedupe_key_from_row(row),
                ]
            )


def merge_policy(user_policy: dict, base_policy: dict) -> dict:
    merged = dict(base_policy)
    for key, value in user_policy.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_policy(value, merged[key])
        else:
            merged[key] = value
    return merged


def load_policy(policy_path: Path | None) -> dict:
    policy = json.loads(json.dumps(DEFAULT_POLICY))
    if policy_path is None or not policy_path.exists():
        return policy

    loaded = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Policy file must contain a JSON object: {policy_path}")
    return merge_policy(loaded, policy)


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
        aliased_pairs = material_strength_exposure.get(alias, [])
        if material == "Stahlbeton":
            # Lean strengths are not valid for reinforced concrete
            aliased_pairs = [(s, e) for s, e in aliased_pairs if s not in LEAN_STRENGTHS]
        return aliased_pairs
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


def _lookup_entity_materials_any_predefined(
    entity: str,
    entity_material: dict[tuple[str, str], list[str]],
) -> list[str]:
    """Collect all materials configured for an entity across predefined variants."""
    collected = []
    for (mapped_entity, _mapped_predefined_type), materials in entity_material.items():
        if mapped_entity != entity:
            continue
        collected.extend(materials)
    return unique_preserve_order(collected)


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
    policy: dict | None = None,
):
    raw_rows: list[QueryRow] = []
    skipped_without_material = 0
    materials_without_strength = 0

    for entity, predefined_types in entity_predefined.items():
        # Always include a variant without predefined type for each entity.
        predefined_values = unique_preserve_order([*(predefined_types or []), DEFAULT_VALUE])
        entity_diam = entity_diameters.get(entity, [""])
        if not entity_diam:
            entity_diam = [""]

        for predefined_type in predefined_values:
            materials = _lookup_materials(entity, predefined_type, entity_material)
            if not materials and predefined_type == DEFAULT_VALUE:
                # If no explicit default exists, fallback to the union of all entity materials.
                materials = _lookup_entity_materials_any_predefined(entity, entity_material)
            is_reinforcement_entity = entity.casefold() in {
                "ifcreinforcingbar",
                "ifcreinforcingmesh",
            }
            if not materials and not is_reinforcement_entity:
                skipped_without_material += 1
                continue
            if not materials:
                materials = [DEFAULT_VALUE]

            for material in materials:
                if not material and is_reinforcement_entity:
                    pairs = _lookup_strength_exposure("Betonstahl", material_strength_exposure)
                else:
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
                    # Lean concrete strengths are restricted to specific entities only
                    if strength in LEAN_STRENGTHS and entity not in LEAN_STRENGTH_ENTITIES:
                        continue
                    # Lean concrete strengths are always in-situ; block PRECAST
                    effective_casting_methods = (
                        ("INSITU",) if strength in LEAN_STRENGTHS and is_concrete_family
                        else casting_methods
                    )
                    for diameter in diameters:
                        for casting_method in effective_casting_methods:
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

    generated_rows: list[QueryRow] = []
    for row in raw_rows:
        slots = extract_semantic_slots(row)
        _apply_casting_alias(slots, policy)
        generated_rows.extend(generate_canonical_queries(slots))
        generated_rows.extend(generate_variant_queries(slots, policy=policy))
        generated_rows.extend(
            generate_confusion_targeted_queries(
                slots,
                confusion_profile=(policy or {}).get("confusion_expansions"),
            )
        )

    unique_rows = deduplicate_and_balance(generated_rows, policy=policy)

    stats = {
        "raw_rows": len(raw_rows),
        "unique_rows": len(unique_rows),
        "skipped_without_material": skipped_without_material,
        "materials_without_strength": materials_without_strength,
    }
    return unique_rows, stats


def write_queries(path: Path, rows: list[tuple[str, str, str, str, str, str, str]]):
    export_query_artifacts(path, rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SBERT-style query lines from possible*.txt files"
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent.parent)
    parser.add_argument(
        "--entity-predefined-file",
        type=Path,
        default=Path("query_generation/sources/possible_entities-predefinedtypes.txt"),
    )
    parser.add_argument(
        "--entity-material-file",
        type=Path,
        default=Path("query_generation/sources/possible_entity-material.txt"),
    )
    parser.add_argument(
        "--entity-diameter-file",
        type=Path,
        default=Path("query_generation/sources/possible_entity-durchmesser.txt"),
    )
    parser.add_argument(
        "--material-strength-exposure-file",
        type=Path,
        default=Path("query_generation/sources/possible_material-strength_exposure.txt"),
    )
    parser.add_argument(
        "--output-file-without-exposure",
        type=Path,
        default=Path("query_generation/generated_queries/generated_queries.txt"),
    )
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=Path("query_generation/query_generation_policy.json"),
    )
    parser.add_argument(
        "--write-debug-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    output_without_exposure_path = resolve_path(base_dir, args.output_file_without_exposure)
    policy_path = resolve_path(base_dir, args.policy_file)

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
            "Create the missing possible_*.txt files in the query_generation/sources folder."
        )

    entity_predefined, warn_ep = parse_entity_predefined(entity_predefined_path)
    entity_material, warn_em = parse_entity_material(entity_material_path)
    entity_diameter, warn_ed = parse_entity_diameters(entity_diameter_path)
    material_strength_exposure, warn_mse = parse_material_strength_exposure(
        material_strength_exposure_path
    )
    policy = load_policy(policy_path)

    all_warnings = warn_ep + warn_em + warn_ed + warn_mse

    rows_without_exposure, stats_without_exposure = generate_queries(
        entity_predefined=entity_predefined,
        entity_material=entity_material,
        material_strength_exposure=material_strength_exposure,
        entity_diameters=entity_diameter,
        include_exposure=False,
        policy=policy,
    )

    output_without_exposure_path.parent.mkdir(parents=True, exist_ok=True)
    write_queries(output_without_exposure_path, rows_without_exposure)

    debug_enabled = bool(policy.get("debug_exports", {}).get("enabled", False))
    if args.write_debug_artifacts and debug_enabled:
        export_debug_artifacts(output_without_exposure_path, rows_without_exposure)

    print(
        "[OK] Wrote "
        f"{stats_without_exposure['unique_rows']} unique queries WITHOUT exposure to "
        f"{output_without_exposure_path}"
    )
    print(f"[INFO] Raw rows before dedup (without exposure): {stats_without_exposure['raw_rows']}")
    print(
        "[INFO] Skipped (entity, predefined_type) without material mapping: "
        f"{stats_without_exposure['skipped_without_material']}"
    )
    print(
        "[INFO] Material combinations without strength/exposure (only entity+material): "
        f"{stats_without_exposure['materials_without_strength']}"
    )
    if all_warnings:
        print(f"[WARN] {len(all_warnings)} parse warning(s):")
        for warning in all_warnings[:20]:
            print(f"  - {warning}")
        if len(all_warnings) > 20:
            print(f"  ... and {len(all_warnings) - 20} more")


if __name__ == "__main__":
    main()