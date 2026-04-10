#!/usr/bin/env python3
"""
Regenerate mapping_generated_queries.txt with improved mapping
rules for better balanced material representation for BGE-M3 fine-tuning.
"""

from pathlib import Path
import subprocess
import csv
import json

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_QUERIES_DIR = SCRIPT_DIR / "generated_queries"
QUERIES_FILE = OUTPUT_QUERIES_DIR / "generated_queries.txt"
MAPPING_FILE = OUTPUT_QUERIES_DIR / "mapping_generated_queries.txt"
POLICY_FILE = SCRIPT_DIR / "query_generation_policy.json"
REPO_ROOT = SCRIPT_DIR.parent.parent

# Solid/sawn timber set aligned with eval expectations for structural wood members
MASSIVHOLZ_SET = (
    "Konstruktionsvollholz | "
    "Massivholz Buche Eiche kammergetrocknet gehobelt | "
    "Massivholz Buche Eiche kammergetrocknet rau | "
    "Massivholz Buche Eiche luftgetrocknet rau | "
    "Massivholz Fichte Tanne Lärche kammergetrocknet gehobelt | "
    "Massivholz Fichte Tanne Lärche luftgetrocknet gehobelt | "
    "Massivholz Fichte Tanne Lärche luftgetrocknet rau"
)

# Species-specific subsets of MASSIVHOLZ_SET
MASSIVHOLZ_BUCHE_EICHE = (
    "Massivholz Buche Eiche kammergetrocknet gehobelt | "
    "Massivholz Buche Eiche kammergetrocknet rau | "
    "Massivholz Buche Eiche luftgetrocknet rau"
)

MASSIVHOLZ_FICHTE_TANNE = (
    "Konstruktionsvollholz | "
    "Massivholz Fichte Tanne Lärche kammergetrocknet gehobelt | "
    "Massivholz Fichte Tanne Lärche luftgetrocknet gehobelt | "
    "Massivholz Fichte Tanne Lärche luftgetrocknet rau"
)


DEFAULT_MAPPING_POLICY = {
    "mapping": {
        "steel_grade_tokens": ["S235", "S235JR", "S235J0", "S355", "S355JR", "S355J0", "S460"],
        "plastic_alias_tokens": ["KUNSTSTOFF", "PTFE", "TEFLON", "POLYTETRAFLUOROETHYLENE"],
        "aggregate_tokens": ["KIES", "SCHOTTER", "GESTEIN", "NATURSTEIN", "AGGREGATE"],
        "pavement_tokens": ["PAVEMENT"],
        "npk_tokens": ["NPK"],
        "npk_implies_insitu": True,
        "write_debug_csv": True,
        "debug_csv_file": "debug/mapping_generated_queries_debug.csv",
    }
}


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


def load_mapping_policy(policy_path: Path) -> dict:
    policy = json.loads(json.dumps(DEFAULT_MAPPING_POLICY))
    if not policy_path.exists():
        return policy

    loaded = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Policy file must contain a JSON object: {policy_path}")
    return merge_policy(loaded, policy)


def _contains_any(haystack_upper: str, needles: list[str]) -> bool:
    return any(needle and needle.upper() in haystack_upper for needle in needles)


def get_diameter(parts: list[str]) -> int | None:
    """Extract pile diameter (numeric token) from query parts."""
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def pile_beton_insitu(
    diameter: int | None,
    displacement: bool = False,
    include_base: bool = True,
) -> str:
    """Build INSITU pile concrete mapping based on diameter and type.

    Parameters
    ----------
    include_base:
        When True (default) prepend ``Bohrpfahlbeton | `` so that generic
        IfcPile queries without a specific predefined type carry the generic
        material alongside the size-specific one.  Set to False for
        BORED/COHESION/JETGROUTING/SUPPORT where the predefined type already
        implies a bored pile — training on the specific size variant only
        avoids the model defaulting to the overly generic Bohrpfahlbeton.
    """
    if displacement:
        base = "Tiefbaubeton"
        if diameter is not None and diameter >= 560:
            deep = "Tiefgründung Ortbetonverdrängungspfahl 660/580"
        else:
            deep = "Tiefgründung Ortbetonverdrängungspfahl 560/480"
        return f"{base} | {deep}"

    if diameter is not None:
        if diameter <= 300:
            deep = "Tiefgründung Mikrobohrpfahl"
        elif diameter <= 700:
            deep = "Tiefgründung Ortbetonbohrpfahl 700"
        elif diameter <= 900:
            deep = "Tiefgründung Ortbetonbohrpfahl 900"
        else:
            deep = "Tiefgründung Ortbetonbohrpfahl 1200"
    else:
        deep = "Tiefgründung Ortbetonbohrpfahl 700"

    if include_base:
        return f"Bohrpfahlbeton | {deep}"
    return deep


def precast_beton_by_strength(has_normal_strength: bool, has_high_strength: bool) -> str:
    """Map PRECAST concrete to normal/high strength classes based on query grade."""
    if has_normal_strength and not has_high_strength:
        return "Betonfertigteil normalfest"
    if has_high_strength and not has_normal_strength:
        return "Betonfertigteil hochfest"
    return "Betonfertigteil normalfest | Betonfertigteil hochfest"


def get_mapping(query: str, original: str, policy: dict) -> str:
    """Return improved mapping for a query line."""
    parts = query.split()
    if not parts:
        return original

    mapping_policy = policy.get("mapping", {})
    steel_grade_tokens = [str(token).upper() for token in mapping_policy.get("steel_grade_tokens", [])]
    plastic_alias_tokens = [str(token).upper() for token in mapping_policy.get("plastic_alias_tokens", [])]
    aggregate_tokens = [str(token).upper() for token in mapping_policy.get("aggregate_tokens", [])]
    pavement_tokens = [str(token).upper() for token in mapping_policy.get("pavement_tokens", [])]
    npk_tokens = [str(token).upper() for token in mapping_policy.get("npk_tokens", [])]
    npk_implies_insitu = bool(mapping_policy.get("npk_implies_insitu", True))

    query_upper = query.upper()

    entity = parts[0]
    predefined = parts[1] if len(parts) > 1 else ""

    has_beton = "BETON" in query_upper
    has_magerbeton = "MAGERBETON" in query_upper
    has_polyolefin = "POLYOLEFIN" in query_upper
    has_npk = _contains_any(query_upper, npk_tokens)
    has_precast = "PRECAST" in query_upper or "FERTIGTEIL" in query_upper
    # Explicit PRECAST must win over any implicit INSITU inference.
    has_insitu = ("INSITU" in query_upper or "ORTBETON" in query_upper) and not has_precast
    if (not has_insitu) and (not has_precast) and npk_implies_insitu and has_npk:
        has_insitu = True
    has_stahl = _contains_any(query_upper, steel_grade_tokens) or (
        "STAHL" in query_upper and "STAHLBETON" not in query_upper
    )
    has_verguetungsstahl = "VERGÜTUNGSSTAHL" in query_upper or "VERGUETUNGSSTAHL" in query_upper
    if has_verguetungsstahl:
        has_stahl = True
    has_litze = "LITZE" in query_upper
    has_spannstahl = "SPANNSTAHL" in query_upper or "DRAHT" in query_upper
    has_holz = "Holz" in query
    has_buche = "Buche" in query
    has_eiche = "Eiche" in query
    has_fichte = "Fichte" in query
    has_tanne = "Tanne" in query
    has_laerche = "LÄRCHE" in query_upper
    has_aluminium = "Aluminium" in query
    has_metal = "Metall" in query
    has_kunststoff = _contains_any(query_upper, plastic_alias_tokens)
    has_naturstein = "Naturstein" in query or "AGGREGATE" in query_upper
    has_gestein = "Gestein" in query
    has_schotter = "Schotter" in query
    has_erdmaterial = "Erdmaterial" in query
    has_mineralisch_filter = "Mineralischer Filter" in query or "Mineralischer_Filter" in query
    has_kies_sand_gemisch = "Kies-Sand-Gemisch" in query or "Kies_Sand_Gemisch" in query
    has_mauerwerk = "Mauerwerk" in query
    has_feinmoertel = "Feinmörtel" in query
    has_asphalt = (
        "Asphalt" in query
        or "Walzasphalt" in query
        or "Splittmastixasphalt" in query
        or "Offenporiger Asphalt" in query
        or "Offenporiger_Asphalt" in query
        or _contains_any(query_upper, pavement_tokens)
    )
    has_bitumen = "Bitumen" in query
    has_bitumenmischgut = "Bitumenmischgut" in query
    has_polymod_bitumen = "Polymermodifiziertes Bitumen" in query or "Polymermodifiziertes_Bitumen" in query
    has_hydraulisch_tragschicht = "Hydraulisch" in query and "Tragschicht" in query
    has_ungebundene_tragschicht = "Ungebundene" in query and "Tragschicht" in query
    has_hydraulisch_fundament = "Hydraulisch" in query and "Fundament" in query
    has_elastomer = "ELASTOMER" in query_upper
    has_kies = ("Kies" in query) or _contains_any(query_upper, aggregate_tokens)
    has_normal_strength = "C20/25" in query or "C25/30" in query or "C30/37" in query
    has_high_strength = "C35/45" in query or "C40/50" in query
    has_lean_strength = "C12/15" in query or "C16/20" in query
    diameter = get_diameter(parts)

    # ── Early escapes for unambiguous material tokens ─────────────
    if has_magerbeton:
        return "Magerbeton"
    if has_beton and has_lean_strength:
        return "Magerbeton"

    # ── IfcPile ──────────────────────────────────────────────────────
    if entity == "IfcPile":
        if has_kies:
            return "Tiefgründung Rüttelstopfsäule"
        if predefined == "DRIVEN":
            if has_stahl:
                return "Stahlblech blank | Stahlprofil blank"
            if has_precast:
                return "Tiefgründung Vorgefertigter Betonpfahl"
            if has_beton:
                return "Tiefgründung Vorgefertigter Betonpfahl"

        if predefined == "SUPPORT":
            if has_stahl:
                return (
                    "Tiefgründung Mikrobohrpfahl | "
                    "Stahlblech blank | Stahlprofil blank"
                )
            if has_kies:
                return (
                    "Tiefgründung Rüttelstopfsäule"
                )
            if has_insitu:
                return pile_beton_insitu(diameter, include_base=False)
            if has_precast:
                return "Tiefgründung Vorgefertigter Betonpfahl"

        if predefined == "FRICTION":
            if has_insitu:
                return pile_beton_insitu(diameter, displacement=True)
            if has_precast:
                return "Tiefgründung Vorgefertigter Betonpfahl"

        # BORED, COHESION, JETGROUTING, and any other specific predefined type:
        # include_base=False so the model learns the concrete size-specific
        # variant rather than the generic Bohrpfahlbeton.
        if has_insitu:
            return pile_beton_insitu(diameter, include_base=False)
        if has_precast:
            return "Tiefgründung Vorgefertigter Betonpfahl"
        if has_beton:
            return (
                f"{pile_beton_insitu(diameter)} | "
                "Tiefgründung Vorgefertigter Betonpfahl"
            )

    # ── IfcReinforcingBar / IfcReinforcingMesh / IfcTendon ────────────
    if entity in ("IfcReinforcingBar", "IfcReinforcingMesh", "IfcTendon"):
        if has_litze or has_spannstahl:
            return "Armierungsstahl"
        return "Armierungsstahl"

    # ── IfcTendon / IfcTendonAnchor ──────────────────────────────────
    if entity in ("IfcTendonAnchor"):
        if has_stahl:
            return "Stahlblech blank | Stahlprofil blank | Stahlblech verzinkt"

    # ── IfcTendonConduit ─────────────────────────────────────────────
    if entity == "IfcTendonConduit":
        if predefined in ("DUCT", "GROUTING_DUCT"):
            if has_kunststoff:
                return "Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC"
            if has_stahl:
                return "Stahlblech blank | Stahlprofil blank"
        elif predefined in ("COUPLER", "DIABOLO"):
            if has_kunststoff:
                return (
                    "Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC | "
                    "Polycarbonat PC | Polyamid PA glasfaserverstärkt"
                )
            if has_stahl:
                return "Stahlblech blank | Stahlprofil blank"
        elif has_stahl:
            return "Stahlblech blank | Stahlprofil blank"

    # ── IfcBearing ───────────────────────────────────────────────────
    if entity == "IfcBearing":
        if has_kunststoff or has_elastomer:
            return "Polyurethan PUR PIR | Kautschukdichtungsmasse | Dichtungsbahn Gummi EPDM"
        if has_stahl:
            return "Stahlblech verzinkt"

    # ── IfcPipeFitting / IfcPipeSegment ──────────────────────────────
    if entity in ("IfcPipeFitting", "IfcPipeSegment"):
        if has_kunststoff:
            return "Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC"
        if has_stahl:
            return "Stahlblech blank | Stahlprofil blank | Gusseisen"

    # ── IfcDiscreteAccessory ─────────────────────────────────────────
    if entity == "IfcDiscreteAccessory":
        if predefined == "EXPANSION_JOINT_DEVICE":
            if has_kunststoff:
                return "Kautschukdichtungsmasse | Polysulfiddichtungsmasse | Silicon-Fugenmasse"
            if has_stahl:
                return "Stahlblech blank | Stahlblech verzinkt | Stahlprofil blank"
        if predefined == "ELASTIC_CUSHION":
            if has_kunststoff:
                return "Polyurethan PUR PIR | Polyethylen PE"
        if predefined == "RAILPAD":
            if has_kunststoff:
                return "Polyamid PA glasfaserverstärkt | Polyethylen PE | Polypropylen PP | Polyurethan PUR PIR"
        if predefined == "INSULATOR":
            if has_kunststoff:
                return "Polyamid PA glasfaserverstärkt | Polycarbonat PC"
        if predefined == "BIRDPROTECTION":
            if has_kunststoff:
                return "Polycarbonat PC | Polypropylen PP | Polyvinylchlorid PVC"
        if predefined == "FILLER":
            if has_kunststoff:
                return "Polystyrol expandiert EPS | Polystyrol extrudiert XPS | Polyurethan PUR PIR | Polyethylen PE"
        if predefined == "FLASHING":
            if has_stahl:
                return "Stahlblech blank | Stahlblech verzinkt"
        if predefined == "CABLEARRANGER":
            if has_kunststoff:
                return "Acrylnitril-Butadien-Styrol ABS | Polyamid PA glasfaserverstärkt | Polypropylen PP"
        if predefined == "SOUNDABSORPTION":
            if has_kunststoff:
                return "Polystyrol expandiert EPS | Polyurethan PUR PIR | Polypropylen PP"

    # ── IfcCovering ──────────────────────────────────────────────────
    if entity == "IfcCovering":
        if predefined == "MEMBRANE":
            if has_bitumen:
                return "Dichtungsbahn bituminös"
            if has_kunststoff:
                return (
                    "Dichtungsbahn Gummi EPDM | Dichtungsbahn Polyolefin FPO | "
                    "Polyethylenfolie PE | Polyethylenvlies PE"
                )
        if predefined == "CLADDING":
            if has_buche or has_eiche:
                return MASSIVHOLZ_BUCHE_EICHE
            if has_fichte or has_tanne or has_laerche:
                return MASSIVHOLZ_FICHTE_TANNE
            if has_holz:
                return (
                    "Furniersperrholz | Holzwolle-Leichtbauplatte zementgebunden | "
                    "Konstruktionsvollholz | "
                    "Massivholz Buche Eiche kammergetrocknet gehobelt | "
                    "Massivholz Fichte Tanne Lärche kammergetrocknet gehobelt"
                )
            if has_naturstein:
                return (
                    "Natursteinplatte poliert | Natursteinplatte geschliffen | "
                    "Natursteinplatte geschnitten | Hartsandsteinplatte | Kalksteinplatte | Kunststeinplatte zementgebunden"
                )
            if has_kunststoff:
                return "Polyvinylchlorid PVC | Polypropylen PP | Acrylnitril-Butadien-Styrol ABS | Plexiglas PMMA Acrylglas"
            if has_metal:
                return "Gusseisen | Stahlblech blank | Stahlblech verzinkt"
        if predefined == "COPING":
            if has_naturstein:
                return (
                    "Natursteinplatte geschnitten | Natursteinplatte geschliffen | "
                    "Natursteinplatte poliert | Hartsandsteinplatte | Kalksteinplatte | "
                    "Kalksandstein | Kunststeinplatte zementgebunden"
                )
            if has_aluminium:
                return "Aluminiumblech blank | Kupferblech blank | Aluminiumprofil blank"
            if has_metal:
                return "Chromnickelstahlblech blank | Chromnickelstahlblech verzinnt | Chromstahlblech blank | Chromstahlblech verzinnt | Kupferblech blank | Messing-/Baubronzeblech | Blei | Gusseisen"
            
        if predefined == "MOLDING":
            if has_buche or has_eiche:
                return MASSIVHOLZ_BUCHE_EICHE
            if has_fichte or has_tanne or has_laerche:
                return MASSIVHOLZ_FICHTE_TANNE
            if has_holz:
                return (
                    "Konstruktionsvollholz | "
                    "Massivholz Buche Eiche kammergetrocknet gehobelt | "
                    "Massivholz Fichte Tanne Lärche kammergetrocknet gehobelt"
                )
            if has_kunststoff:
                return "Polystyrol expandiert EPS | Acrylnitril-Butadien-Styrol ABS | Polyvinylchlorid PVC"
        if predefined == "TOPPING":
            if has_feinmoertel:
                return (
                    "Baukleber Einbettmörtel mineralisch | "
                    "Hartbeton einschichtig | Hartbeton zweischichtig"
                )
        if predefined == "WRAPPING":
            if has_kunststoff:
                return "Polyethylenfolie PE | Polyethylenvlies PE | Polyethylen PE"
        # Fallback when no predefined type matches
        if has_feinmoertel:
            return (
                "Baukleber Einbettmörtel mineralisch | "
                "Hartbeton einschichtig | Hartbeton zweischichtig"
            )

    # ── IfcCourse ────────────────────────────────────────────────────
    if entity == "IfcCourse":
        if has_erdmaterial:
            return "Sand | Rundkies | Kies gebrochen"
        if has_mineralisch_filter:
            return "Kies gebrochen | Sand"
        if has_kies_sand_gemisch:
            return "Rundkies | Sand"
        if has_schotter:
            return "Kies gebrochen"
        if predefined == "ARMOUR":
            if has_gestein:
                return "Kalksteinplatte | Hartsandsteinplatte | Kies gebrochen"
            if has_naturstein:
                return "Kies gebrochen | Rundkies"
        if predefined == "BALLASTBED":
            if has_kies:
                return "Rundkies"
            if has_gestein or has_naturstein:
                return "Kies gebrochen | Rundkies"
        if predefined == "CORE":
            if has_gestein:
                return "Kies gebrochen"
        if predefined == "FILTER":
            if has_naturstein:
                return "Kies gebrochen | Rundkies | Sand"
        if predefined == "PAVEMENT":
            if has_beton:
                return "Hartbeton einschichtig | Hartbeton zweischichtig | Tiefbaubeton"
            if has_asphalt:
                return "Gussasphalt"
            if has_bitumen:
                return "Gussasphalt | Dichtungsbahn bituminös"
        if predefined == "PROTECTION":
            if has_gestein:
                return "Kies gebrochen | Rundkies"
            if has_naturstein:
                return "Kies gebrochen | Rundkies | Natursteinplatte geschnitten"
        if has_beton and has_insitu:
            return "Hartbeton einschichtig | Hartbeton zweischichtig | Tiefbaubeton"
        if has_beton and has_precast:
            return precast_beton_by_strength(has_normal_strength, has_high_strength)
        if has_beton:
            return (
                "Hartbeton einschichtig | Hartbeton zweischichtig | Tiefbaubeton | "
                f"{precast_beton_by_strength(has_normal_strength, has_high_strength)}"
            )

    # ── IfcFooting ───────────────────────────────────────────────────
    if entity == "IfcFooting":
        mager_excluded = has_normal_strength or has_high_strength or has_npk or "STAHLBETON" in query_upper
        if has_beton and has_insitu:
            if mager_excluded:
                return "Tiefbaubeton"
            return "Tiefbaubeton | Magerbeton"
        if has_beton and has_precast:
            return precast_beton_by_strength(has_normal_strength, has_high_strength)
        if has_beton:
            if mager_excluded:
                return (
                    f"Tiefbaubeton | "
                    f"{precast_beton_by_strength(has_normal_strength, has_high_strength)}"
                )
            return (
                f"Tiefbaubeton | Magerbeton | "
                f"{precast_beton_by_strength(has_normal_strength, has_high_strength)}"
            )

    # ── IfcKerb ──────────────────────────────────────────────────────
    if entity == "IfcKerb":
        if has_naturstein:
            return "Hartsandsteinplatte | Kalksteinplatte | Natursteinplatte geschnitten | Kunststeinplatte zementgebunden"

    # ── IfcWall ──────────────────────────────────────────────────────
    if entity == "IfcWall":
        if predefined == "RETAININGWALL":
            if has_naturstein:
                return (
                    "Natursteinplatte geschnitten | Natursteinplatte geschliffen | "
                    "Hartsandsteinplatte | Kalksteinplatte | Kunststeinplatte zementgebunden"
                )
            if has_stahl:
                return (
                    "Baugrubensicherung Spundwand auskragend | "
                    "Baugrubensicherung Spundwand gespriesst | "
                    "Baugrubensicherung Spundwand verankert | "
                    "Stahlblech blank | Stahlprofil blank"
                )
            if has_beton and has_insitu:
                return (
                    "Tiefbaubeton | "
                    "Baugrubensicherung Bohrpfahlwand verankert | "
                    "Baugrubensicherung Bohrpfahlwand unverankert | "
                    "Baugrubensicherung Bohrpfahlwand gespriesst | "
                    "Baugrubensicherung Schlitzwand 400 | Baugrubensicherung Schlitzwand 800 | "
                    "Baugrubensicherung Nagelwand"
                )
            if has_beton and has_precast:
                return precast_beton_by_strength(has_normal_strength, has_high_strength)
            if has_beton:
                return (
                    "Tiefbaubeton | "
                    f"{precast_beton_by_strength(has_normal_strength, has_high_strength)} | "
                    "Baugrubensicherung Bohrpfahlwand verankert | "
                    "Baugrubensicherung Bohrpfahlwand unverankert | "
                    "Baugrubensicherung Bohrpfahlwand gespriesst | "
                    "Baugrubensicherung Schlitzwand 400 | Baugrubensicherung Schlitzwand 800 | "
                    "Baugrubensicherung Nagelwand"
                )
            return (
                "Baugrubensicherung Spundwand auskragend | "
                "Baugrubensicherung Spundwand gespriesst | "
                "Baugrubensicherung Spundwand verankert | "
                "Baugrubensicherung Bohrpfahlwand verankert | "
                "Baugrubensicherung Bohrpfahlwand unverankert | "
                "Baugrubensicherung Bohrpfahlwand gespriesst | "
                "Baugrubensicherung Rühlwand auskragend | "
                "Baugrubensicherung Rühlwand gespriesst | "
                "Baugrubensicherung Rühlwand verankert | "
                "Baugrubensicherung Schlitzwand 400 | Baugrubensicherung Schlitzwand 800 | "
                "Baugrubensicherung Nagelwand"
            )
        if predefined == "WAVEWALL":
            if has_stahl:
                return (
                    "Baugrubensicherung Spundwand auskragend | "
                    "Baugrubensicherung Spundwand gespriesst | "
                    "Baugrubensicherung Spundwand verankert | "
                    "Stahlblech blank | Stahlprofil blank"
                )
        if predefined in ("PARAPET", "POLYGONAL"):
            if has_naturstein:
                return "Natursteinplatte geschnitten | Hartsandsteinplatte | Kalksteinplatte | Kunststeinplatte zementgebunden"
            if has_mauerwerk:
                return "Backstein | Kalksandstein | Betonziegel"
        _IFCWALL_KNOWN_PREDEFINED = {
            "RETAININGWALL", "WAVEWALL", "PARAPET", "POLYGONAL",
            "STANDARD", "SOLIDWALL", "SHEAR", "ELEMENTEDWALL",
            "MOVABLE", "PLUMBINGWALL",
        }
        _no_predefined = (
            predefined == ""
            or predefined == "STANDARD"
            or predefined.upper() not in _IFCWALL_KNOWN_PREDEFINED
        )
        if _no_predefined:
            if has_stahl:
                return (
                    "Baugrubensicherung Spundwand auskragend | "
                    "Baugrubensicherung Spundwand gespriesst | "
                    "Baugrubensicherung Spundwand verankert | "
                    "Stahlblech blank | Stahlprofil blank"
                )
            if has_beton and has_insitu:
                return (
                    "Tiefbaubeton | "
                    "Baugrubensicherung Bohrpfahlwand verankert | "
                    "Baugrubensicherung Bohrpfahlwand unverankert | "
                    "Baugrubensicherung Bohrpfahlwand gespriesst | "
                    "Baugrubensicherung Rühlwand auskragend | "
                    "Baugrubensicherung Rühlwand gespriesst | "
                    "Baugrubensicherung Rühlwand verankert | "
                    "Baugrubensicherung Schlitzwand 400 | Baugrubensicherung Schlitzwand 800 | "
                    "Baugrubensicherung Nagelwand"
                )
            if has_beton and has_precast:
                return precast_beton_by_strength(has_normal_strength, has_high_strength)
            if has_beton:
                return (
                    "Tiefbaubeton | "
                    f"{precast_beton_by_strength(has_normal_strength, has_high_strength)} | "
                    "Baugrubensicherung Bohrpfahlwand verankert | "
                    "Baugrubensicherung Bohrpfahlwand unverankert | "
                    "Baugrubensicherung Bohrpfahlwand gespriesst | "
                    "Baugrubensicherung Rühlwand auskragend | "
                    "Baugrubensicherung Rühlwand gespriesst | "
                    "Baugrubensicherung Rühlwand verankert | "
                    "Baugrubensicherung Schlitzwand 400 | Baugrubensicherung Schlitzwand 800 | "
                    "Baugrubensicherung Nagelwand"
                )
        if predefined in ("STANDARD", "SOLIDWALL"):
            if has_buche or has_eiche:
                return MASSIVHOLZ_BUCHE_EICHE
            if has_fichte or has_tanne or has_laerche:
                return MASSIVHOLZ_FICHTE_TANNE
            if has_holz:
                return "3- und 5-Schicht Massivholzplatte | Brettsperrholz | Balkenschichtholz"
            if has_naturstein:
                return (
                    "Natursteinplatte geschnitten | Natursteinplatte geschliffen | "
                    "Hartsandsteinplatte | Kalksteinplatte | Kalksandstein | Kunststeinplatte zementgebunden"
                )
            if has_mauerwerk:
                return "Backstein | Kalksandstein | Betonziegel"
        if has_beton and has_insitu:
            return "Tiefbaubeton"
        if has_beton and has_precast:
            return precast_beton_by_strength(has_normal_strength, has_high_strength)

    # ── IfcSlab ──────────────────────────────────────────────────────
    if entity == "IfcSlab":
        if predefined in ("FLOOR", "ROOF"):
            if has_buche or has_eiche:
                return MASSIVHOLZ_BUCHE_EICHE
            if has_fichte or has_tanne or has_laerche:
                return MASSIVHOLZ_FICHTE_TANNE
            if has_holz:
                return "3- und 5-Schicht Massivholzplatte | Brettsperrholz | Balkenschichtholz | Brettschichtholz"
            if predefined == "FLOOR" and has_insitu and has_high_strength:
                return (
                    "Tiefbaubeton | Hartbeton einschichtig | Hartbeton zweischichtig | "
                    "2K-Fliessbelag Epoxidharz"
                )
        if predefined == "BASESLAB":
            if has_beton and has_insitu:
                if has_normal_strength or has_high_strength or has_npk or "STAHLBETON" in query_upper:
                    return "Tiefbaubeton"
                return "Tiefbaubeton | Magerbeton"
        if predefined in ("PAVING", "SIDEWALK", "WEARING"):
            if has_asphalt:
                return "Gussasphalt"
            if has_bitumen:
                return "Gussasphalt | Dichtungsbahn bituminös"
        if has_beton and has_insitu:
            return "Tiefbaubeton"
        if has_beton and has_precast:
            return precast_beton_by_strength(has_normal_strength, has_high_strength)

    # ── IfcMember ────────────────────────────────────────────────────
    if entity == "IfcMember":
        if predefined == "STAY_CABLE":
            if has_stahl or has_litze:
                return "Armierungsstahl"
        if predefined == "STRUCTURALCABLE":
            if has_stahl or has_litze:
                return "Stahlprofil blank | Armierungsstahl"
        if predefined == "SUSPENDER":
            if has_litze:
                return "Stahlprofil blank | Armierungsstahl"
        if predefined == "SUSPENSION_CABLE":
            if has_stahl or has_litze:
                return "Stahlprofil blank | Armierungsstahl"
        if has_litze:
            return "Armierungsstahl"
        if has_buche or has_eiche:
            return MASSIVHOLZ_BUCHE_EICHE
        if has_fichte or has_tanne or has_laerche:
            return MASSIVHOLZ_FICHTE_TANNE
        if has_holz:
            return MASSIVHOLZ_SET

    # ── IfcTrackElement ──────────────────────────────────────────────
    if entity == "IfcTrackElement":
        if predefined in ("FROG", "HALF_SET_OF_BLADES"):
            if has_verguetungsstahl:
                return "Stahlprofil blank"
            if has_stahl:
                return "Stahlprofil blank"
        if predefined == "SLEEPER":
            if has_polyolefin:
                return "Polyethylen PE | Polypropylen PP"
            if has_buche or has_eiche:
                return MASSIVHOLZ_BUCHE_EICHE
            if has_holz:
                return MASSIVHOLZ_BUCHE_EICHE
            if has_kunststoff:
                return (
                    "Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC | "
                    "Polycarbonat PC | Polyamid PA glasfaserverstärkt"
                )
        if has_kunststoff:
            return (
                "Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC | "
                "Polycarbonat PC | Polyamid PA glasfaserverstärkt"
            )
        if has_holz:
            return MASSIVHOLZ_BUCHE_EICHE

    # ── IfcPlate ─────────────────────────────────────────────────────
    if entity == "IfcPlate":
        if has_stahl:
            return "Stahlblech blank | Stahlblech verzinkt"
        if has_aluminium:
            return "Aluminiumblech blank"

    # ── IfcRail ──────────────────────────────────────────────────────
    if entity == "IfcRail":
        if has_stahl:
            return "Stahlprofil blank"

    # ── IfcRailing ───────────────────────────────────────────────────
    if entity == "IfcRailing":
        if has_buche or has_eiche:
            return MASSIVHOLZ_BUCHE_EICHE
        if has_fichte or has_tanne or has_laerche:
            return MASSIVHOLZ_FICHTE_TANNE
        if has_holz:
            return MASSIVHOLZ_SET
        if has_aluminium:
            return "Aluminiumprofil blank"
        if has_kunststoff:
            return "Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC"

    # ── IfcPavement ──────────────────────────────────────────────────
    if entity == "IfcPavement":
        if has_ungebundene_tragschicht:
            return "Kies gebrochen"
        if has_hydraulisch_tragschicht or has_hydraulisch_fundament:
            return "Hartbeton einschichtig | Hartbeton zweischichtig"
        if predefined == "FLEXIBLE":
            if has_bitumenmischgut or has_polymod_bitumen:
                return "Gussasphalt"
            if has_asphalt:
                return "Gussasphalt"
            if has_bitumen:
                return "Gussasphalt | Dichtungsbahn bituminös"
        if predefined == "RIGID":
            if has_beton:
                return "Hartbeton einschichtig | Hartbeton zweischichtig"

    # ── IfcBeam ──────────────────────────────────────────────────────
    if entity == "IfcBeam":
        if has_stahl:
            return "Stahlprofil blank"

    # ── Generic fallback rules ───────────────────────────────────────
    if has_litze or has_spannstahl:
        return "Armierungsstahl"
    if has_beton and has_insitu:
        return "Tiefbaubeton"
    if has_beton and has_precast:
        return precast_beton_by_strength(has_normal_strength, has_high_strength)
    if has_beton:
        return f"Tiefbaubeton | {precast_beton_by_strength(has_normal_strength, has_high_strength)}"
    if has_stahl:
        return "Stahlblech blank | Stahlprofil blank"
    if has_buche or has_eiche:
        return MASSIVHOLZ_BUCHE_EICHE
    if has_fichte or has_tanne or has_laerche:
        return MASSIVHOLZ_FICHTE_TANNE
    if has_holz:
        return "Balkenschichtholz | Brettschichtholz | Brettsperrholz | Konstruktionsvollholz"
    if has_aluminium:
        return "Aluminiumblech blank | Aluminiumprofil blank"
    if has_kunststoff:
        return "Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC | Polyurethan PUR PIR"
    if has_naturstein:
        return "Natursteinplatte geschnitten | Natursteinplatte geschliffen | Hartsandsteinplatte | Kalksteinplatte | Kunststeinplatte zementgebunden"
    if has_mauerwerk:
        return "Backstein | Kalksandstein | Betonziegel"
    if has_polyolefin:
        return "Dichtungsbahn Polyolefin FPO"
    if has_asphalt:
        return "Gussasphalt"
    if has_bitumen:
        return "Dichtungsbahn bituminös"
    if has_metal:
        return "Chromnickelstahlblech blank | Chromnickelstahlblech verzinnt | Chromstahlblech blank | Chromstahlblech verzinnt | Kupferblech blank | Messing-/Baubronzeblech | Blei | Gusseisen"
    if has_kies:
        return "Kies gebrochen | Rundkies"

    return original


def git_show_lines(repo_relative_path: str) -> list[str] | None:
    """Read file contents from git HEAD and return split lines."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{repo_relative_path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        return None
    return result.stdout.splitlines()


def main() -> None:
    query_lines = QUERIES_FILE.read_text(encoding="utf-8").splitlines()
    policy = load_mapping_policy(POLICY_FILE)
    head_query_lines = git_show_lines(
        "Training/query_generation/generated_queries/generated_queries.txt"
    )
    head_mapping_lines = git_show_lines(
        "Training/query_generation/generated_queries/mapping_generated_queries_without_exposure.txt"
    )

    baseline_by_query: dict[str, str] = {}
    baseline_line_count = 0

    # Prefer HEAD query->mapping pairs so reordering/new queries do not shift fallback lines.
    if (
        head_query_lines is not None
        and head_mapping_lines is not None
        and len(head_query_lines) == len(head_mapping_lines)
    ):
        baseline_by_query = {
            q.strip(): m.strip()
            for q, m in zip(head_query_lines, head_mapping_lines)
            if q.strip()
        }
        baseline_line_count = len(head_mapping_lines)
        print(f"Using original mapping from HEAD ({baseline_line_count} lines)")
    else:
        current_mapping_lines = (
            MAPPING_FILE.read_text(encoding="utf-8").splitlines()
            if MAPPING_FILE.exists()
            else []
        )
        baseline_by_query = {
            q.strip(): m.strip()
            for q, m in zip(query_lines, current_mapping_lines)
            if q.strip()
        }
        baseline_line_count = len(current_mapping_lines)
        print(f"Using current mapping file ({baseline_line_count} lines)")

    if len(query_lines) != baseline_line_count:
        print(
            "Line count differs "
            f"({len(query_lines)} queries vs {baseline_line_count} baseline mappings); "
            "regenerating and overwriting mapping file from current queries."
        )

    new_mappings: list[str] = []
    debug_rows: list[list[str | int]] = []
    changed = 0
    mapping_policy = policy.get("mapping", {})
    unresolved_queries: list[str] = []
    for line_number, q in enumerate(query_lines, start=1):
        query = q.strip()
        baseline = baseline_by_query.get(query, "")
        if baseline == "UNMAPPED":
            baseline = ""
        new_m = get_mapping(query, baseline, policy=policy)

        # Enforce complete coverage: every query must resolve to a concrete mapping.
        if new_m == "UNMAPPED" or not str(new_m).strip():
            unresolved_queries.append(query)
            continue

        if new_m != baseline:
            changed += 1
        new_mappings.append(new_m)
        debug_rows.append(
            [
                line_number,
                query,
                baseline,
                new_m,
                "changed" if new_m != baseline else "unchanged",
                "non-empty",
            ]
        )

    if unresolved_queries:
        examples = "\n".join(f"  - {query}" for query in unresolved_queries[:20])
        raise RuntimeError(
            "Found queries without mapping. Please extend mapping rules before exporting.\n"
            f"Count: {len(unresolved_queries)}\n"
            f"Examples:\n{examples}"
        )

    output = "\n".join(new_mappings)
    MAPPING_FILE.write_text(output, encoding="utf-8")

    if bool(mapping_policy.get("write_debug_csv", True)):
        debug_file_name = str(
            mapping_policy.get(
                "debug_csv_file",
                "debug/mapping_generated_queries_debug.csv",
            )
        ).strip() or "debug/mapping_generated_queries_debug.csv"
        debug_path = OUTPUT_QUERIES_DIR / debug_file_name
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "line_number",
                    "query",
                    "baseline_mapping",
                    "new_mapping",
                    "change_state",
                    "mapping_state",
                ]
            )
            writer.writerows(debug_rows)

    print(
        f"Done. {changed}/{len(query_lines)} lines updated. "
        f"Wrote {len(new_mappings)} lines."
    )


if __name__ == "__main__":
    main()
