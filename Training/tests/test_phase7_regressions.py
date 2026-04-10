import json
import tempfile
import types
import unittest
from pathlib import Path

from Training import run_training_pipeline as pipeline
from Training import train_bge_m3 as trainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestPhase7Regressions(unittest.TestCase):
    def test_compute_strict_hn_viability_reports_unusable_without_hn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pairs_file = Path(tmp_dir) / "pairs_no_hn.jsonl"
            rows = [{"query": f"q{i}", "positive": f"p{i}"} for i in range(8)]
            write_jsonl(pairs_file, rows)

            result = pipeline.compute_strict_hn_viability(
                pairs_file=pairs_file,
                dev_ratio=0.2,
                seed=42,
                hard_negative_selection="first",
                num_hard_negatives=1,
            )

        self.assertFalse(result["strict_hn_usable"])
        self.assertIn("strict", result["strict_error"].lower())
        self.assertEqual(result["train_queries_with_hn_in_trainer_input"], 0)

    def test_read_records_accepts_string_list_and_none_hn_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pairs_file = Path(tmp_dir) / "mixed_hn_schema.jsonl"
            rows = [
                {"query": "q1", "positive": "p1", "hard_negatives": "hn1"},
                {"query": "q2", "positive": "p2", "hard_negatives": ["hn2", "hn2", "p2", "", "hn3"]},
                {"query": "q3", "positive": "p3", "hard_negatives": None},
            ]
            write_jsonl(pairs_file, rows)

            records = trainer.read_records(pairs_file)

        self.assertEqual(records[0].hard_negatives, ("hn1",))
        self.assertEqual(records[1].hard_negatives, ("hn2", "hn3"))
        self.assertEqual(records[2].hard_negatives, ())

    def test_pipeline_coerce_str_list_handles_schema_variants(self) -> None:
        self.assertEqual(pipeline.coerce_str_list(None), [])
        self.assertEqual(pipeline.coerce_str_list(" hn "), ["hn"])
        self.assertEqual(pipeline.coerce_str_list(["a", " ", "b"]), ["a", "b"])
        self.assertEqual(pipeline.coerce_str_list(123), [])

    def test_unique_positive_sampler_respects_query_positive_union(self) -> None:
        examples = [
            types.SimpleNamespace(texts=["qA", "p1"]),
            types.SimpleNamespace(texts=["qB", "p2"]),
            types.SimpleNamespace(texts=["qC", "p3"]),
            types.SimpleNamespace(texts=["qD", "p4"]),
        ]
        query_positive_union = {
            "qA": {"p1", "p2"},
            "qB": {"p2"},
            "qC": {"p3"},
            "qD": {"p4"},
        }

        sampler = trainer.UniquePositiveBatchSampler(
            examples=examples,
            batch_size=2,
            seed=7,
            query_positive_union=query_positive_union,
        )

        normalized_union = {
            trainer.normalize_text_key(query): {trainer.normalize_text_key(pos) for pos in positives}
            for query, positives in query_positive_union.items()
        }

        for batch in sampler:
            batch_positives = [trainer.normalize_text_key(examples[index].texts[1]) for index in batch]
            self.assertEqual(len(batch_positives), len(set(batch_positives)))

            for left_index in batch:
                left_query = trainer.normalize_text_key(examples[left_index].texts[0])
                known_positives = normalized_union.get(left_query, set())
                for right_index in batch:
                    if left_index == right_index:
                        continue
                    right_positive = trainer.normalize_text_key(examples[right_index].texts[1])
                    self.assertNotIn(right_positive, known_positives)

    def test_unique_positive_sampler_avoids_duplicate_positive_in_batch(self) -> None:
        examples = [
            types.SimpleNamespace(texts=["q1", "pX"]),
            types.SimpleNamespace(texts=["q2", "pX"]),
            types.SimpleNamespace(texts=["q3", "pY"]),
            types.SimpleNamespace(texts=["q4", "pZ"]),
        ]

        sampler = trainer.UniquePositiveBatchSampler(
            examples=examples,
            batch_size=2,
            seed=11,
        )

        for batch in sampler:
            batch_positives = [trainer.normalize_text_key(examples[index].texts[1]) for index in batch]
            self.assertEqual(len(batch_positives), len(set(batch_positives)))

    def test_resolve_prefix_strategy_defaults_to_no_prefix(self) -> None:
        default_strategy = pipeline.resolve_prefix_strategy(
            run_instruction_ab=False,
            legacy_query_prefix="",
        )
        legacy_strategy = pipeline.resolve_prefix_strategy(
            run_instruction_ab=True,
            legacy_query_prefix="Instruction: ",
        )

        self.assertEqual(default_strategy["prefix_mode"], "no_prefix")
        self.assertEqual(default_strategy["source_of_prefix_setting"], "dense_only_bge_m3_default")
        self.assertTrue(default_strategy["dense_only_bge_m3_default_applied"])
        self.assertFalse(default_strategy["legacy_prefix_experiment_active"])

        self.assertEqual(legacy_strategy["prefix_mode"], "legacy_prefix")
        self.assertFalse(legacy_strategy["dense_only_bge_m3_default_applied"])
        self.assertTrue(legacy_strategy["legacy_prefix_experiment_active"])

    def test_split_record_keys_digest_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pairs_file = Path(tmp_dir) / "pairs_for_split.jsonl"
            rows = []
            for idx in range(12):
                rows.append({"query": f"query {idx}", "positive": f"positive {idx}"})
            write_jsonl(pairs_file, rows)

            train_a, dev_a = pipeline.split_record_keys_from_pairs(pairs_file, dev_ratio=0.25, seed=99)
            train_b, dev_b = pipeline.split_record_keys_from_pairs(pairs_file, dev_ratio=0.25, seed=99)

        self.assertEqual(train_a, train_b)
        self.assertEqual(dev_a, dev_b)
        self.assertEqual(pipeline.record_keys_sha1(train_a), pipeline.record_keys_sha1(train_b))
        self.assertEqual(pipeline.record_keys_sha1(dev_a), pipeline.record_keys_sha1(dev_b))


if __name__ == "__main__":
    unittest.main()
