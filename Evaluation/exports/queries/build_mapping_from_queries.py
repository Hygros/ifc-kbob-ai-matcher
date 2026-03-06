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


def pile_beton_insitu(diameter: int | None, add_arma: bool, displacement: bool = False) -> str:
    """Build INSITU pile concrete mapping based on diameter and type."""
    if displacement:
        base = "Tiefbaubeton"
        if diameter is not None and diameter >= 560:
            deep = "Tiefgründung Ortbetonverdrängungspfahl 660/580"
        else:
            deep = "Tiefgründung Ortbetonverdrängungspfahl 560/480"
        result = f"{base} | {deep}"
        if add_arma:
            result += " | Armierungsstahl"
        return result

    base = "Tiefbaubeton | Bohrpfahlbeton"
    if diameter is not None:
        if diameter <= 200:
            deep = "Tiefgründung Ortbetonbohrpfahl 700 | Tiefgründung Mikrobohrpfahl"
        elif diameter <= 700:
            deep = "Tiefgründung Ortbetonbohrpfahl 700"
        elif diameter <= 900:
            deep = "Tiefgründung Ortbetonbohrpfahl 900"
        else:
            deep = "Tiefgründung Ortbetonbohrpfahl 1200"
        result = f"{base} | {deep}"
    else:
        result = f"{base} | Tiefgründung Ortbetonbohrpfahl 700"
    if add_arma:
        result += " | Armierungsstahl"
    return result


def get_mapping(query: str, original: str) -> str:
    """Return improved mapping for a query line."""
    parts = query.split()
    if not parts:
        return original

    entity = parts[0]
    predefined = parts[1] if len(parts) > 1 else ""

    has_stahlbeton = "Stahlbeton" in query
    has_beton = "Beton" in query and not has_stahlbeton
    has_insitu = "INSITU" in query
    has_precast = "PRECAST" in query
    has_stahl = any(s in query for s in ["Stahl S235", "Stahl S355", "Stahl S460"])
    has_holz = "Holz" in query
    has_aluminium = "Aluminium" in query
    has_kunststoff = "Kunststoff" in query
    has_naturstein = "Naturstein" in query
    has_mauerwerk = "Mauerwerk" in query
    has_asphalt = "Asphalt" in query
    has_bitumen = "Bitumen" in query
    has_betonstahl = "Betonstahl" in query
    has_high_strength = "C35/45" in query or "C40/50" in query
    diameter = get_diameter(parts)

    # ── IfcPile ──────────────────────────────────────────────────────
    if entity == "IfcPile":
        if has_betonstahl:
            return "Armierungsstahl"

        if predefined == "DRIVEN":
            if has_stahl:
                return (
                    "Baugrubensicherung Spundwand auskragend | "
                    "Baugrubensicherung Spundwand gespriesst | "
                    "Baugrubensicherung Spundwand verankert | "
                    "Stahlblech blank | Stahlprofil blank"
                )
            if has_precast:
                base = "Betonfertigteil normalfest | Betonfertigteil hochfest | Tiefgründung Vorgefertigter Betonpfahl"
                return base + " | Armierungsstahl" if has_stahlbeton else base

        if predefined == "SUPPORT":
            if has_stahl:
                return (
                    "Baugrubensicherung Rühlwand auskragend | "
                    "Baugrubensicherung Rühlwand gespriesst | "
                    "Baugrubensicherung Rühlwand verankert | "
                    "Baugrubensicherung Nagelwand | "
                    "Tiefgründung Mikrobohrpfahl | "
                    "Stahlblech blank | Stahlprofil blank"
                )
            if has_insitu:
                return pile_beton_insitu(diameter, has_stahlbeton)
            if has_precast:
                base = "Tiefgründung Vorgefertigter Betonpfahl"
                return base + " | Armierungsstahl" if has_stahlbeton else base

        if predefined == "FRICTION":
            if has_insitu:
                return pile_beton_insitu(diameter, has_stahlbeton, displacement=True)
            if has_precast:
                base = "Tiefgründung Vorgefertigter Betonpfahl"
                return base + " | Armierungsstahl" if has_stahlbeton else base

        # BORED, COHESION, JETGROUTING, etc.
        if has_insitu:
            return pile_beton_insitu(diameter, has_stahlbeton)
        if has_precast:
            base = "Tiefgründung Vorgefertigter Betonpfahl"
            return base + " | Armierungsstahl" if has_stahlbeton else base

    # ── IfcReinforcingBar / IfcReinforcingMesh ───────────────────────
    if entity in ("IfcReinforcingBar", "IfcReinforcingMesh"):
        return "Armierungsstahl"

    # ── IfcTendon / IfcTendonAnchor ──────────────────────────────────
    if entity in ("IfcTendon", "IfcTendonAnchor"):
        if has_stahl:
            return "Armierungsstahl"

    # ── IfcTendonConduit ─────────────────────────────────────────────
    if entity == "IfcTendonConduit":
        if predefined in ("DUCT", "GROUTING_DUCT"):
            if has_kunststoff:
                return "Polyethylen PE | Polypropylen PP | Polyvinylchlorid PVC"
            if has_stahl:
                return "Armierungsstahl | Stahlblech blank | Stahlprofil blank"
        elif has_stahl:
            return "Armierungsstahl | Stahlblech blank | Stahlprofil blank"

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
                    "Natursteinplatte geschnitten | Hartsandsteinplatte | Kalksteinplatte"
                )
            if has_kunststoff:
                return "Polyvinylchlorid PVC | Polypropylen PP | Acrylnitril-Butadien-Styrol ABS | Plexiglas PMMA Acrylglas"
        if predefined == "COPING":
            if has_naturstein:
                return (
                    "Natursteinplatte geschnitten | Natursteinplatte geschliffen | "
                    "Hartsandsteinplatte | Kalksteinplatte"
                )
            if has_aluminium:
                return "Aluminiumblech blank | Kupferblech blank | Aluminiumprofil blank"
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
            return "Hartsandsteinplatte | Kalksteinplatte | Natursteinplatte geschnitten"

    # ── IfcWall ──────────────────────────────────────────────────────
    if entity == "IfcWall":
        if predefined == "RETAININGWALL":
            if has_naturstein:
                return (
                    "Natursteinplatte geschnitten | Natursteinplatte geschliffen | "
                    "Hartsandsteinplatte | Kalksteinplatte"
                )
            if has_insitu:
                if has_stahlbeton:
                    return (
                        "Tiefbaubeton | Armierungsstahl | "
                        "Baugrubensicherung Schlitzwand 400 | Baugrubensicherung Schlitzwand 800 | "
                        "Baugrubensicherung Bohrpfahlwand verankert | "
                        "Baugrubensicherung Bohrpfahlwand unverankert | "
                        "Baugrubensicherung Bohrpfahlwand gespriesst"
                    )
                if has_beton:
                    return (
                        "Tiefbaubeton | "
                        "Baugrubensicherung Schlitzwand 400 | Baugrubensicherung Schlitzwand 800"
                    )
        if predefined in ("PARAPET", "POLYGONAL"):
            if has_naturstein:
                return "Natursteinplatte geschnitten | Hartsandsteinplatte | Kalksteinplatte"
            if has_mauerwerk:
                return "Backstein | Kalksandstein | Betonziegel"
        if predefined in ("STANDARD", "SOLIDWALL"):
            if has_holz:
                return "3- und 5-Schicht Massivholzplatte | Brettsperrholz | Balkenschichtholz"
            if has_naturstein:
                return (
                    "Natursteinplatte geschnitten | Natursteinplatte geschliffen | "
                    "Hartsandsteinplatte | Kalksteinplatte | Kalksandstein"
                )
            if has_mauerwerk:
                return "Backstein | Kalksandstein | Betonziegel"
        if has_stahlbeton and has_insitu:
            return "Tiefbaubeton | Armierungsstahl"
        if has_stahlbeton and has_precast:
            return "Betonfertigteil normalfest | Betonfertigteil hochfest | Armierungsstahl"

    # ── IfcSlab ──────────────────────────────────────────────────────
    if entity == "IfcSlab":
        if predefined in ("FLOOR", "ROOF"):
            if has_holz:
                return "3- und 5-Schicht Massivholzplatte | Brettsperrholz | Balkenschichtholz | Brettschichtholz"
            if predefined == "FLOOR" and has_insitu:
                if has_high_strength and not has_stahlbeton:
                    # High-strength floor concrete → can be Hartbeton or 2K-Fliessbelag in industrial contexts
                    return (
                        "Tiefbaubeton | Hartbeton einschichtig | Hartbeton zweischichtig | "
                        "2K-Fliessbelag Epoxidharz"
                    )
                if has_high_strength and has_stahlbeton:
                    return (
                        "Tiefbaubeton | Armierungsstahl | "
                        "Hartbeton einschichtig | Hartbeton zweischichtig"
                    )
        if predefined == "BASESLAB":
            if has_beton and has_insitu:
                return "Tiefbaubeton | Magerbeton"
            if has_stahlbeton and has_insitu:
                return "Tiefbaubeton | Magerbeton | Armierungsstahl"
        if predefined in ("PAVING", "SIDEWALK", "WEARING"):
            if has_asphalt:
                return "Gussasphalt | Heissbitumen"
        if has_stahlbeton and has_insitu:
            return "Tiefbaubeton | Armierungsstahl"
        if has_stahlbeton and has_precast:
            return "Betonfertigteil normalfest | Betonfertigteil hochfest | Armierungsstahl"

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

    # ── Generic fallback rules for Stahlbeton ────────────────────────
    if has_stahlbeton and has_insitu:
        return "Tiefbaubeton | Armierungsstahl"
    if has_stahlbeton and has_precast:
        return "Betonfertigteil normalfest | Betonfertigteil hochfest | Armierungsstahl"

    return original


def main() -> None:
    query_lines = QUERIES_FILE.read_text(encoding="utf-8").splitlines()
    # Use original git HEAD mapping as baseline
    result = subprocess.run(
        ["git", "show", "HEAD:Evaluation/exports/queries/mapping_generated_queries_without_exposure.txt"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    if result.returncode == 0:
        original_lines = result.stdout.splitlines()
        print(f"Using original mapping from HEAD ({len(original_lines)} lines)")
    else:
        original_lines = MAPPING_FILE.read_text(encoding="utf-8").splitlines()
        print(f"Using current mapping file ({len(original_lines)} lines)")

    if len(query_lines) != len(original_lines):
        raise ValueError(
            f"Line count mismatch: {len(query_lines)} queries vs {len(original_lines)} mappings"
        )

    new_mappings: list[str] = []
    changed = 0
    for q, m in zip(query_lines, original_lines):
        new_m = get_mapping(q.strip(), m.strip())
        if new_m != m.strip():
            changed += 1
        new_mappings.append(new_m)

    output = "\n".join(new_mappings) + "\n"
    MAPPING_FILE.write_text(output, encoding="utf-8")
    print(f"Done. {changed}/{len(query_lines)} lines updated.")


if __name__ == "__main__":
    main()
