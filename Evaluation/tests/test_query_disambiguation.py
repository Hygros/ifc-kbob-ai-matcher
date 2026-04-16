import unittest

from Dashboard.services.training_export import record_to_query
from core.sbert.sentence_transformer import extract_disambiguation_tokens, ifc_entry_to_string


class TestQueryDisambiguation(unittest.TestCase):
    def test_matches_canonical_baugrubensicherung_tokens(self):
        self.assertEqual(
            extract_disambiguation_tokens("Baugrubensicherung Bohrpfahlwand verankert", ifc_entity="IfcWall"),
            ["Bohrpfahlwand", "verankert"],
        )
        self.assertEqual(
            extract_disambiguation_tokens("Baugrubensicherung Schlitzwand 800", ifc_entity="IfcWall"),
            ["Schlitzwand", "800"],
        )

    def test_handles_typos(self):
        self.assertEqual(
            extract_disambiguation_tokens("Schlizwand D800", ifc_entity="IfcWall"),
            ["Schlitzwand", "800"],
        )
        self.assertEqual(
            extract_disambiguation_tokens("Bohrpfalwand unverankert", ifc_entity="IfcWall"),
            ["Bohrpfahlwand", "unverankert"],
        )

    def test_handles_multilingual_name_tokens(self):
        self.assertEqual(
            extract_disambiguation_tokens("paroi moulee ancre", ifc_entity="IfcWall"),
            ["Schlitzwand", "verankert"],
        )
        self.assertEqual(
            extract_disambiguation_tokens("soil nail wall braced", ifc_entity="IfcWall"),
            ["Nagelwand", "gespriesst"],
        )

    def test_ignores_irrelevant_names(self):
        self.assertEqual(extract_disambiguation_tokens("Flügelwand", ifc_entity="IfcWall"), [])
        self.assertEqual(extract_disambiguation_tokens("Belag Holz", ifc_entity="IfcWall"), [])

    def test_disambiguation_is_disabled_for_non_wall_entities(self):
        self.assertEqual(
            extract_disambiguation_tokens("Bohrpfahl", ifc_entity="IfcPile"),
            [],
        )
        pile_entry = {
            "IfcEntity": "IfcPile",
            "PredefinedType": "BORED",
            "Material": "Stahlbeton",
            "Name": "Bohrpfahl",
        }
        self.assertEqual(ifc_entry_to_string(pile_entry), "IfcPile BORED Stahlbeton")
        self.assertEqual(record_to_query(pile_entry), "IfcPile BORED Stahlbeton")

    def test_sbert_and_dashboard_build_same_query(self):
        entry = {
            "IfcEntity": "IfcWall",
            "PredefinedType": "RETAININGWALL",
            "Material": "Stahlbeton",
            "Name": "Schlizwand D800",
        }
        self.assertEqual(ifc_entry_to_string(entry), record_to_query(entry))
        self.assertEqual(ifc_entry_to_string(entry), "IfcWall RETAININGWALL Stahlbeton Schlitzwand 800")


if __name__ == "__main__":
    unittest.main()
