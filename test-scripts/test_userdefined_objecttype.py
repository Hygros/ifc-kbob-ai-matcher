import argparse
from pathlib import Path

import ifcopenshell


def iter_userdefined_elements(model):
    for element in model.by_type("IfcElement"):
        predefined_type = getattr(element, "PredefinedType", None)
        if str(predefined_type).upper() == "USERDEFINED":
            yield element


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print ObjectType for IFC elements with PredefinedType=USERDEFINED"
    )
    parser.add_argument("ifc_file", help="Path to IFC file")
    args = parser.parse_args()

    ifc_path = Path(args.ifc_file)
    if not ifc_path.exists():
        print(f"IFC file not found: {ifc_path}")
        return 1

    model = ifcopenshell.open(str(ifc_path))

    count = 0
    for element in iter_userdefined_elements(model):
        count += 1
        entity = element.is_a() if hasattr(element, "is_a") else "<unknown>"
        guid = getattr(element, "GlobalId", "") or ""
        name = getattr(element, "Name", "") or ""
        object_type = getattr(element, "ObjectType", None)
        print(
            f"{count:04d} | Entity={entity} | GUID={guid} | Name={name} | ObjectType={object_type}"
        )

    if count == 0:
        print("No IfcElement found with PredefinedType=USERDEFINED")
    else:
        print(f"Found {count} element(s) with PredefinedType=USERDEFINED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
