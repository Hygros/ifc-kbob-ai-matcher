#!/usr/bin/env python3
"""
Regenerate mapping_generated_queries_without_exposure.txt with improved mapping
rules for better balanced material representation for BGE-M3 fine-tuning.
"""

from pathlib import Path
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent
QUERIES_FILE = SCRIPT_DIR / "generated_queries_without_exposure.txt"
MAPPING_FILE = SCRIPT_DIR / "mapping_generated_queries_without_exposure.txt"
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def get_diameter(parts: list[str]) -> int | None:
    """Extract pile diameter (numeric token) from query parts."""
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def pile_beton_insitu(diameter: int | None, displacement: bool = False) -> str:
    """Build INSITU pile concrete mapping based on diameter and type."""
    if displacement:
        base = "Tiefbaubeton"
        if diameter is not None and diameter >= 560:
            deep = "Tiefgründung Ortbetonverdrängungspfahl 660/580"
        else:
            deep = "Tiefgründung Ortbetonverdrängungspfahl 560/480"
        return f"{base} | {deep}"

    base = "Bohrpfahlbeton"
    if diameter is not None:
        if diameter <= 300:
            deep = "Tiefgründung Mikrobohrpfahl"
        elif diameter <= 700:
            deep = "Tiefgründung Ortbetonbohrpfahl 700"
        elif diameter <= 900:
            deep = "Tiefgründung Ortbetonbohrpfahl 900"
        else:
            deep = "Tiefgründung Ortbetonbohrpfahl 1200"
        return f"{base} | {deep}"
    return f"{base} | Tiefgründung Ortbetonbohrpfahl 700"


def precast_beton_by_strength(has_normal_strength: bool, has_high_strength: bool) -> str:
    """Map PRECAST concrete to normal/high strength classes based on query grade."""
    if has_normal_strength and not has_high_strength:
        return "Betonfertigteil normalfest"
    if has_high_strength and not has_normal_strength:
        return "Betonfertigteil hochfest"
    return "Betonfertigteil normalfest | Betonfertigteil hochfest"


def get_mapping(query: str, original: str) -> str:
    """Return improved mapping for a query line."""
    parts = query.split()
    if not parts:
        return original

    entity = parts[0]
    predefined = parts[1] if len(parts) > 1 else ""

    has_beton = "Beton" in query or "Stahlbeton" in query
    has_insitu = "INSITU" in query
    has_precast = "PRECAST" in query
    has_stahl = any(s in query for s in ["Stahl S235", "Stahl S355", "Stahl S460"])
    has_holz = "Holz" in query
    has_aluminium = "Aluminium" in query
    has_metal = "Metall" in query
    has_kunststoff = "Kunststoff" in query
    has_naturstein = "Naturstein" in query
    has_mauerwerk = "Mauerwerk" in query
    has_asphalt = "Asphalt" in query
    has_bitumen = "Bitumen" in query
    has_kies = "Kies" in query
    has_normal_strength = "C25/30" in query or "C30/37" in query
    has_high_strength = "C35/45" in query or "C40/50" in query
    diameter = get_diameter(parts)

    # ── IfcPile ──────────────────────────────────────────────────────
    if entity == "IfcPile":
        if has_kies:
            return "Tiefgründung Rüttelstopfsäule"
        if predefined == "DRIVEN":
            if has_stahl:
                return (
                    "Baugrubensicherung Spundwand auskragend | "
                    "Baugrubensicherung Spundwand gespriesst | "
                    "Baugrubensicherung Spundwand verankert | "
                    "Stahlblech blank | Stahlprofil blank"
                )
            if has_precast:
                return (
                    f"{precast_beton_by_strength(has_normal_strength, has_high_strength)} | "
                    "Tiefgründung Vorgefertigter Betonpfahl"
                )

        if predefined == "SUPPORT":
            if has_stahl:
                return (
                    "Baugrubensicherung Rühlwand auskragend | "
                    "Baugrubensicherung Rühlwand gespriesst | "
                    "Baugrubensicherung Rühlwand verankert | "
                    "Tiefgründung Mikrobohrpfahl | "
                    "Stahlblech blank | Stahlprofil blank"
                )
            if has_kies:
                return (
                    "Tiefgründung Rüttelstopfsäule"
                )
            if has_insitu:
                return pile_beton_insitu(diameter)
            if has_precast:
                return "Tiefgründung Vorgefertigter Betonpfahl"

        if predefined == "FRICTION":
            if has_insitu:
                return pile_beton_insitu(diameter, displacement=True)
            if has_precast:
                return "Tiefgründung Vorgefertigter Betonpfahl"

        # BORED, COHESION, JETGROUTING, etc.
        if has_insitu:
            return pile_beton_insitu(diameter)
        if has_precast:
            return "Tiefgründung Vorgefertigter Betonpfahl"

    # ── IfcReinforcingBar / IfcReinforcingMesh ───────────────────────
    if entity in ("IfcReinforcingBar", "IfcReinforcingMesh", "IfcTendon"):
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
        elif has_stahl:
            return "Stahlblech blank | Stahlprofil blank"

    # ── IfcBearing ───────────────────────────────────────────────────
    if entity == "IfcBearing":
        if predefined in ("DISK", "ELASTOMERIC", "POT"):
            if has_kunststoff:
                return "Polyurethan PUR PIR | Polyethylen PE | Polypropylen PP"
        if has_kunststoff:
            return "Acrylnitril-Butadien-Styrol ABS | Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC"
        if has_stahl:
            return "Stahlblech blank | Stahlprofil blank"

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
                return "Dichtungsbahn bituminös | Heissbitumen | Bitumenemulsion"
            if has_kunststoff:
                return (
                    "Dichtungsbahn Gummi EPDM | Dichtungsbahn Polyolefin FPO | "
                    "Polyethylenfolie PE | Polyethylenvlies PE"
                )
        if predefined == "CLADDING":
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
        if predefined == "COPING":
            if has_naturstein:
                return (
                    "Natursteinplatte geschnitten | Natursteinplatte geschliffen | "
                    "Hartsandsteinplatte | Kalksteinplatte | Kunststeinplatte zementgebunden"
                )
            if has_aluminium:
                return "Aluminiumblech blank | Kupferblech blank | Aluminiumprofil blank"
            if has_metal:
                return "Chromnickelstahlblech blank | Chromnickelstahlblech verzinnt | Chromstahlblech blank | Chromstahlblech verzinnt | Kupferblech blank | Messing-/Baubronzeblech | Blei | Gusseisen"
            
        if predefined == "MOLDING":
            if has_holz:
                return (
                    "Konstruktionsvollholz | "
                    "Massivholz Buche Eiche kammergetrocknet gehobelt | "
                    "Massivholz Fichte Tanne Lärche kammergetrocknet gehobelt"
                )
            if has_kunststoff:
                return "Polystyrol expandiert EPS | Acrylnitril-Butadien-Styrol ABS | Polyvinylchlorid PVC"

    # ── IfcCourse ────────────────────────────────────────────────────
    if entity == "IfcCourse":
        if predefined == "ARMOUR":
            if has_naturstein:
                return "Kies gebrochen | Rundkies"
        if predefined == "BALLASTBED":
            if has_naturstein:
                return "Kies gebrochen | Rundkies"
        if predefined == "FILTER":
            if has_naturstein:
                return "Kies gebrochen | Rundkies | Sand"
        if predefined == "PROTECTION":
            if has_naturstein:
                return "Kies gebrochen | Rundkies | Natursteinplatte geschnitten"

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
            if has_beton and has_insitu:
                return (
                    "Tiefbaubeton | "
                    "Baugrubensicherung Schlitzwand 400 | Baugrubensicherung Schlitzwand 800 | "
                    "Baugrubensicherung Bohrpfahlwand verankert | "
                    "Baugrubensicherung Bohrpfahlwand unverankert | "
                    "Baugrubensicherung Bohrpfahlwand gespriesst | "
                    "Baugrubensicherung Nagelwand"
                )
        if predefined in ("PARAPET", "POLYGONAL"):
            if has_naturstein:
                return "Natursteinplatte geschnitten | Hartsandsteinplatte | Kalksteinplatte | Kunststeinplatte zementgebunden"
            if has_mauerwerk:
                return "Backstein | Kalksandstein | Betonziegel"
        if predefined in ("STANDARD", "SOLIDWALL"):
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
            if has_holz:
                return "3- und 5-Schicht Massivholzplatte | Brettsperrholz | Balkenschichtholz | Brettschichtholz"
            if predefined == "FLOOR" and has_insitu and has_high_strength:
                return (
                    "Tiefbaubeton | Hartbeton einschichtig | Hartbeton zweischichtig | "
                    "2K-Fliessbelag Epoxidharz"
                )
        if predefined == "BASESLAB":
            if has_beton and has_insitu:
                return "Tiefbaubeton | Magerbeton"
        if predefined in ("PAVING", "SIDEWALK", "WEARING"):
            if has_asphalt:
                return "Gussasphalt | Heissbitumen"
        if has_beton and has_insitu:
            return "Tiefbaubeton"
        if has_beton and has_precast:
            return precast_beton_by_strength(has_normal_strength, has_high_strength)

    # ── IfcTrackElement ──────────────────────────────────────────────
    if entity == "IfcTrackElement":
        if predefined == "SLEEPER":
            if has_holz:
                return (
                    "Konstruktionsvollholz | "
                    "Massivholz Fichte Tanne Lärche luftgetrocknet rau | "
                    "Massivholz Fichte Tanne Lärche kammergetrocknet gehobelt | "
                    "Massivholz Buche Eiche luftgetrocknet rau | "
                    "Massivholz Buche Eiche kammergetrocknet rau"
                )

    # ── IfcRailing ───────────────────────────────────────────────────
    if entity == "IfcRailing":
        if predefined in ("HANDRAIL", "BALUSTRADE"):
            if has_holz:
                return (
                    "Konstruktionsvollholz | "
                    "Massivholz Buche Eiche kammergetrocknet gehobelt | "
                    "Balkenschichtholz"
                )
        if predefined == "FENCE":
            if has_holz:
                return (
                    "Konstruktionsvollholz | "
                    "Massivholz Fichte Tanne Lärche luftgetrocknet gehobelt | "
                    "Massivholz Buche Eiche luftgetrocknet rau | "
                    "Massivholz Buche Eiche kammergetrocknet rau | "
                    "Balkenschichtholz"
                )

    # ── IfcPavement ──────────────────────────────────────────────────
    if entity == "IfcPavement":
        if predefined == "FLEXIBLE":
            if has_asphalt:
                return "Gussasphalt | Heissbitumen | Bitumenemulsion"
            if has_bitumen:
                return "Bitumenemulsion | Heissbitumen | Dichtungsbahn bituminös"

    # ── Generic fallback rules ───────────────────────────────────────
    if has_beton and has_insitu:
        return "Tiefbaubeton"
    if has_beton and has_precast:
        return precast_beton_by_strength(has_normal_strength, has_high_strength)
    if has_stahl:
        return "Stahlblech blank | Stahlprofil blank"
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
    if has_asphalt:
        return "Gussasphalt | Heissbitumen"
    if has_bitumen:
        return "Bitumenemulsion | Heissbitumen | Dichtungsbahn bituminös"
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
    head_query_lines = git_show_lines("Evaluation/exports/queries/generated_queries_without_exposure.txt")
    head_mapping_lines = git_show_lines("Evaluation/exports/queries/mapping_generated_queries_without_exposure.txt")

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
    changed = 0
    empty_lines = 0
    for q in query_lines:
        query = q.strip()
        baseline = baseline_by_query.get(query, "")
        if baseline == "UNMAPPED":
            baseline = ""
        new_m = get_mapping(query, baseline)

        # For unknown cases keep an empty mapping line (same line number as query).
        if new_m == "UNMAPPED":
            new_m = ""
        if not new_m:
            empty_lines += 1

        if new_m != baseline:
            changed += 1
        new_mappings.append(new_m)

    output = "\n".join(new_mappings) + "\n"
    MAPPING_FILE.write_text(output, encoding="utf-8")
    print(
        f"Done. {changed}/{len(query_lines)} lines updated. "
        f"Wrote {len(new_mappings)} lines. Empty mapping lines: {empty_lines}."
    )


if __name__ == "__main__":
    main()
