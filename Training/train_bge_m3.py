import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader, Dataset


class InputExampleDataset(Dataset[InputExample]):
    def __init__(self, examples: list[InputExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> InputExample:
        return self._examples[index]


def read_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Trainingsdatei nicht gefunden: {path}")

    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            query = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            if not query or not positive:
                raise ValueError(f"Ungültiger Datensatz in Zeile {line_no}: query/positive fehlt.")
            pairs.append((query, positive))

    if not pairs:
        raise ValueError("Keine Trainingspaare gefunden.")

    return pairs


def split_pairs(
    pairs: list[tuple[str, str]],
    dev_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio muss im Bereich [0, 1) liegen.")

    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    dev_size = int(len(shuffled) * dev_ratio)
    if dev_ratio > 0 and dev_size == 0 and len(shuffled) > 5:
        dev_size = 1

    dev_pairs = shuffled[:dev_size]
    train_pairs = shuffled[dev_size:]
    if not train_pairs:
        raise ValueError("Nach dem Split sind keine Trainingsdaten übrig. --dev-ratio verringern.")
    return train_pairs, dev_pairs


def build_ir_evaluator(dev_pairs: list[tuple[str, str]]) -> InformationRetrievalEvaluator | None:
    if not dev_pairs:
        return None

    query_to_positives: dict[str, set[str]] = {}
    for query, positive in dev_pairs:
        query_to_positives.setdefault(query, set()).add(positive)

    if not query_to_positives:
        return None

    corpus_docs: dict[str, str] = {}
    doc_id_by_text: dict[str, str] = {}
    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for q_idx, (query, positives) in enumerate(query_to_positives.items()):
        qid = f"q{q_idx}"
        queries[qid] = query
        relevant_docs[qid] = set()

        for positive in positives:
            if positive not in doc_id_by_text:
                doc_id = f"d{len(doc_id_by_text)}"
                doc_id_by_text[positive] = doc_id
                corpus_docs[doc_id] = positive
            relevant_docs[qid].add(doc_id_by_text[positive])

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_docs,
        relevant_docs=relevant_docs,
        name="dev_ir",
    )


def choose_device(user_device: str) -> str:
    user_device = user_device.strip().lower()
    if user_device in {"cpu", "cuda"}:
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning von BAAI/bge-m3 als Bi-Encoder.")
    parser.add_argument("--train-file", required=True, help="Pfad zur JSONL-Datei mit query/positive Paaren.")
    parser.add_argument("--base-model", default="BAAI/bge-m3", help="Sentence-Transformer Startmodell.")
    parser.add_argument("--output-dir", required=True, help="Ausgabeverzeichnis für das trainierte Modell.")
    parser.add_argument("--epochs", type=int, default=2, help="Anzahl Epochen.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch-Size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup-Anteil (0..1).")
    parser.add_argument("--max-length", type=int, default=512, help="Maximale Token-Länge.")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Anteil für Dev-Evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed.")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--fp16", action="store_true", help="Mixed precision Training (use_amp).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs muss > 0 sein.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size muss > 0 sein.")
    if args.lr <= 0:
        raise ValueError("--lr muss > 0 sein.")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio muss im Bereich [0, 1) liegen.")
    if args.max_length <= 0:
        raise ValueError("--max-length muss > 0 sein.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_path = Path(args.train_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = read_pairs(train_path)
    train_pairs, dev_pairs = split_pairs(all_pairs, dev_ratio=args.dev_ratio, seed=args.seed)

    device = choose_device(args.device)
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Gesamtpaare: {len(all_pairs)} | Train: {len(train_pairs)} | Dev: {len(dev_pairs)}")

    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_length

    train_examples = [InputExample(texts=[query, positive]) for query, positive in train_pairs]
    train_dataset = InputExampleDataset(train_examples)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    evaluator = build_ir_evaluator(dev_pairs)
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * args.warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(output_dir),
        save_best_model=evaluator is not None,
        use_amp=args.fp16,
        show_progress_bar=True,
    )

    print(f"Training abgeschlossen. Modell gespeichert unter: {output_dir}")


if __name__ == "__main__":
    main()
