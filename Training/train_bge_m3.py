import argparse
import inspect
import json
import math
import random
import shutil
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
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


class UniquePositiveBatchSampler:
    """BatchSampler ensuring no positive text appears more than once per batch.

    Prevents false negatives in MultipleNegativesRankingLoss by guaranteeing
    that each in-batch negative is a genuinely different document.
    """

    def __init__(self, examples: list[InputExample], batch_size: int, seed: int) -> None:
        self._batch_size = batch_size
        self._seed = seed
        self._epoch = 0

        pos_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, ex in enumerate(examples):
            positive = ex.texts[1] if ex.texts else ""
            pos_to_indices[positive].append(idx)

        self._pos_to_indices = dict(pos_to_indices)
        self._total = len(examples)
        self._unique_positives = len(self._pos_to_indices)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed + self._epoch)
        self._epoch += 1

        group_queues: dict[str, list[int]] = {}
        for pos, indices in self._pos_to_indices.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            group_queues[pos] = shuffled

        batches: list[list[int]] = []

        while group_queues:
            keys = list(group_queues.keys())
            rng.shuffle(keys)

            batch: list[int] = []
            exhausted: list[str] = []

            for key in keys:
                batch.append(group_queues[key].pop())
                if not group_queues[key]:
                    exhausted.append(key)
                if len(batch) == self._batch_size:
                    batches.append(batch)
                    batch = []

            for key in exhausted:
                del group_queues[key]

            if batch:
                batches.append(batch)

        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(self._total / self._batch_size)


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
    """Split on query level: all pairs of a query stay together in train or dev."""
    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio muss im Bereich [0, 1) liegen.")

    if dev_ratio == 0:
        return list(pairs), []

    query_to_pairs: dict[str, list[tuple[str, str]]] = {}
    for query, positive in pairs:
        query_to_pairs.setdefault(query, []).append((query, positive))

    unique_queries = list(query_to_pairs.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_queries)

    dev_query_count = int(len(unique_queries) * dev_ratio)
    if dev_query_count == 0 and len(unique_queries) > 5:
        dev_query_count = 1

    dev_queries = set(unique_queries[:dev_query_count])

    dev_pairs: list[tuple[str, str]] = []
    train_pairs: list[tuple[str, str]] = []
    for query, positive in pairs:
        if query in dev_queries:
            dev_pairs.append((query, positive))
        else:
            train_pairs.append((query, positive))

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
    parser.add_argument(
        "--run-id",
        default="",
        help="Deterministische Run-ID für nachvollziehbare Artefakte (wenn leer, wird ein Fallback verwendet).",
    )
    parser.add_argument(
        "--save-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Speichert zusätzlich zu best-model den Zustand jeder Epoche.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="",
        help="Optionaler Ordner für Epochen-Checkpoints (Default: <output-dir>/epochs).",
    )
    return parser.parse_args()


def sanitize_label(value: str, fallback: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    safe = "".join(ch if ch in allowed else "_" for ch in value).strip("._-")
    return safe or fallback


def normalize_step_checkpoints_to_epochs(step_checkpoint_dir: Path, epoch_checkpoint_dir: Path) -> list[str]:
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not step_checkpoint_dir.is_dir():
        return []

    step_dirs = [path for path in step_checkpoint_dir.iterdir() if path.is_dir()]
    step_dirs.sort(key=lambda item: item.stat().st_mtime)

    saved_epoch_dirs: list[str] = []
    for index, source_dir in enumerate(step_dirs, start=1):
        target_dir = epoch_checkpoint_dir / f"epoch-{index:02d}"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        saved_epoch_dirs.append(str(target_dir))

    return saved_epoch_dirs


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

    run_id = sanitize_label(args.run_id, fallback=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    all_pairs = read_pairs(train_path)
    train_pairs, dev_pairs = split_pairs(all_pairs, dev_ratio=args.dev_ratio, seed=args.seed)

    device = choose_device(args.device)
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    train_queries = len({q for q, _ in train_pairs})
    dev_queries = len({q for q, _ in dev_pairs})
    print(
        f"Gesamtpaare: {len(all_pairs)} | Train: {len(train_pairs)} ({train_queries} Queries) | "
        f"Dev: {len(dev_pairs)} ({dev_queries} Queries)"
    )

    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_length

    train_examples = [InputExample(texts=[query, positive]) for query, positive in train_pairs]
    train_dataset = InputExampleDataset(train_examples)
    batch_sampler = UniquePositiveBatchSampler(train_examples, batch_size=args.batch_size, seed=args.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    # sentence-transformers model.fit() reads dataloader.batch_size internally;
    # PyTorch sets it to None when batch_sampler is used, so patch it back.
    object.__setattr__(train_dataloader, "batch_size", args.batch_size)
    print(f"UniquePositiveBatchSampler: {batch_sampler._unique_positives} unique Positives, batch_size={args.batch_size}")
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    evaluator = build_ir_evaluator(dev_pairs)
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * args.warmup_ratio)

    step_checkpoint_dir: Path | None = None
    epoch_checkpoint_dir: Path | None = None
    steps_per_epoch = max(1, len(train_dataloader))
    checkpoint_args_enabled = False

    if args.save_each_epoch:
        if args.checkpoints_dir:
            epoch_checkpoint_dir = Path(args.checkpoints_dir).expanduser().resolve()
        else:
            epoch_checkpoint_dir = output_dir / "epochs"
        step_checkpoint_dir = output_dir / "_checkpoints_steps"

        fit_params = inspect.signature(model.fit).parameters
        supports_checkpoints = {"checkpoint_path", "checkpoint_save_steps", "checkpoint_save_total_limit"}.issubset(
            set(fit_params.keys())
        )
        if supports_checkpoints:
            checkpoint_args_enabled = True
        else:
            print(
                "Warnung: Diese sentence-transformers-Version unterstützt keine Checkpoint-Argumente in model.fit; "
                "Epochen-Checkpoints werden übersprungen."
            )
            step_checkpoint_dir = None
            epoch_checkpoint_dir = None

    if checkpoint_args_enabled and step_checkpoint_dir is not None:
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
            checkpoint_path=str(step_checkpoint_dir),
            checkpoint_save_steps=int(steps_per_epoch),
            checkpoint_save_total_limit=int(args.epochs),
        )
    else:
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

    epoch_checkpoint_paths: list[str] = []
    if args.save_each_epoch and step_checkpoint_dir is not None and epoch_checkpoint_dir is not None:
        epoch_checkpoint_paths = normalize_step_checkpoints_to_epochs(step_checkpoint_dir, epoch_checkpoint_dir)
        if step_checkpoint_dir.exists():
            shutil.rmtree(step_checkpoint_dir)

    metadata = {
        "run_id": run_id,
        "train_file": str(train_path),
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "device": device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "warmup_ratio": float(args.warmup_ratio),
        "max_length": int(args.max_length),
        "dev_ratio": float(args.dev_ratio),
        "seed": int(args.seed),
        "fp16": bool(args.fp16),
        "save_each_epoch": bool(args.save_each_epoch),
        "steps_per_epoch": int(steps_per_epoch),
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "epoch_checkpoints": epoch_checkpoint_paths,
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Training abgeschlossen. Modell gespeichert unter: {output_dir}")
    if epoch_checkpoint_paths:
        print(f"Epochen-Checkpoints gespeichert unter: {epoch_checkpoint_dir}")
    print(f"Run-Metadaten: {metadata_path}")


if __name__ == "__main__":
    main()
