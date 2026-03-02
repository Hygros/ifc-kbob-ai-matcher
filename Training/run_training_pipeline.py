"""
Für kleine Datensätze (bis ca. 5000 unique-Paare):

Training/run_training_pipeline.py --query-file Evaluation/exports/queries/ifcentity_material_strength.txt --expected-file Evaluation/expected_material/expected.txt --base-model BAAI/bge-m3 --output-dir Training/artifacts/models/bge-m3-finetuned-ifcentity-material-strength-3-epochs --epochs 3 --batch-size 8 --lr 2e-5 --max-length 512 --dev-ratio 0.15 --device auto --deduplicate

epochs 2:
     #Anzahl Trainingsdurchläufe über den Trainingssplit.
batch-size 8:
    Anzahl Trainingspaare pro Schritt; größer = schneller, aber mehr VRAM.
lr 2e-5:
    Learning Rate; bestimmt, wie stark Gewichte pro Update geändert werden.
max-length 512:
    Maximale Tokenlänge pro Text (Query/Positive); längere Texte werden abgeschnitten.
device auto:
    cuda wenn verfügbar, sonst cpu
deduplicate:
    Entfernt identische (query, positive)-Paare, damit doppelte Beispiele das Training nicht verzerren.

Weitere (default):    
warmup-ratio 0.1:
    10% der Trainingsschritte werden als Warmup genutzt (LR steigt zunächst an, stabileres Training).
dev-ratio 0.1:
    10% der Paare gehen in den Dev-Split, 90 % in Train (für Monitoring/Eval während Training).
seed 42:
    Fixiert Zufall (Shuffle/Split) für besser reproduzierbare Runs.
fp16 (Default: aus):
    Mixed Precision ist nur aktiv, wenn du den Flag explizit setzt. Sonst läuft Training in voller Präzision.


usage: run_training_pipeline.py [-h] --query-file QUERY_FILE --expected-file EXPECTED_FILE [--base-model BASE_MODEL]
                                [--pairs-out PAIRS_OUT] [--output-dir OUTPUT_DIR] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
                                [--warmup-ratio WARMUP_RATIO] [--max-length MAX_LENGTH] [--dev-ratio DEV_RATIO] [--seed SEED]
                                [--device DEVICE] [--fp16] [--deduplicate]
run_training_pipeline.py: error: the following arguments are required: --query-file, --expected-file
"""




import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "Training"
VALIDATE_SCRIPT = TRAINING_DIR / "validate_training_data.py"
PREPARE_SCRIPT = TRAINING_DIR / "prepare_training_data.py"
TRAIN_SCRIPT = TRAINING_DIR / "train_bge_m3.py"


def run_command(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Befehl fehlgeschlagen mit Exit-Code {result.returncode}: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestriert die Bi-Encoder-Trainingspipeline: validate -> prepare -> train."
    )
    parser.add_argument("--query-file", required=True, help="Pfad zur Query-TXT.")
    parser.add_argument("--expected-file", required=True, help="Pfad zur Expected-TXT.")
    parser.add_argument("--base-model", default="BAAI/bge-m3", help="Startmodell für Fine-Tuning.")
    parser.add_argument(
        "--pairs-out",
        default="Training/artifacts/training_pairs.jsonl",
        help="Ausgabe-JSONL für Trainingspaare.",
    )
    parser.add_argument(
        "--output-dir",
        default="Training/artifacts/models/bge-m3-finetuned",
        help="Output-Verzeichnis des feinjustierten Modells.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Epochen für Fine-Tuning.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch-Size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup-Anteil.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximale Token-Länge.")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev-Split-Anteil.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed.")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--fp16", action="store_true", help="Mixed precision Training aktivieren.")
    parser.add_argument("--deduplicate", action="store_true", help="Identische query/positive-Paare entfernen.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for script in (VALIDATE_SCRIPT, PREPARE_SCRIPT, TRAIN_SCRIPT):
        if not script.is_file():
            raise FileNotFoundError(f"Skript nicht gefunden: {script}")

    query_file = Path(args.query_file)
    expected_file = Path(args.expected_file)
    pairs_out = Path(args.pairs_out)
    output_dir = Path(args.output_dir)

    run_command(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--query-file",
            str(query_file),
            "--expected-file",
            str(expected_file),
        ]
    )

    prepare_command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--query-file",
        str(query_file),
        "--expected-file",
        str(expected_file),
        "--out",
        str(pairs_out),
    ]
    if args.deduplicate:
        prepare_command.append("--deduplicate")
    run_command(prepare_command)

    run_command([sys.executable, str(VALIDATE_SCRIPT), "--pairs-file", str(pairs_out)])

    train_command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-file",
        str(pairs_out),
        "--base-model",
        args.base_model,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-length",
        str(args.max_length),
        "--dev-ratio",
        str(args.dev_ratio),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.fp16:
        train_command.append("--fp16")

    run_command(train_command)
    print("\nTrainingspipeline abgeschlossen.")


if __name__ == "__main__":
    main()
