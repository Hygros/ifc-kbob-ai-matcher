from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

warnings.warn(
    "[DEPRECATED] run_ablation_suite.py ist als deprecated markiert und wird in einem "
    "zukuenftigen Release entfernt. Nutze run_training_pipeline.py direkt fuer einzelne Laeufe.",
    DeprecationWarning,
    stacklevel=2,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = PROJECT_ROOT / "Training" / "run_training_pipeline.py"


@dataclass(frozen=True)
class AblationConfig:
    config_id: str
    hard_negative_mode: str
    hard_negative_selection: str
    include_fraglich_in_main_training: bool
    weight_eindeutig: float
    weight_mehrdeutig: float
    weight_ungenau: float
    weight_fraglich: float


DEFAULT_ABLATIONS = [
    AblationConfig(
        config_id="ablation_01_hn_off",
        hard_negative_mode="off",
        hard_negative_selection="first",
        include_fraglich_in_main_training=False,
        weight_eindeutig=1.0,
        weight_mehrdeutig=1.0,
        weight_ungenau=0.5,
        weight_fraglich=0.0,
    ),
    AblationConfig(
        config_id="ablation_02_hn_fallback_first",
        hard_negative_mode="fallback",
        hard_negative_selection="first",
        include_fraglich_in_main_training=False,
        weight_eindeutig=1.0,
        weight_mehrdeutig=1.0,
        weight_ungenau=0.5,
        weight_fraglich=0.0,
    ),
    AblationConfig(
        config_id="ablation_03_hn_fallback_random",
        hard_negative_mode="fallback",
        hard_negative_selection="random",
        include_fraglich_in_main_training=False,
        weight_eindeutig=1.0,
        weight_mehrdeutig=1.0,
        weight_ungenau=0.5,
        weight_fraglich=0.0,
    ),
    AblationConfig(
        config_id="ablation_04_hn_strict_first",
        hard_negative_mode="strict",
        hard_negative_selection="first",
        include_fraglich_in_main_training=False,
        weight_eindeutig=1.0,
        weight_mehrdeutig=1.0,
        weight_ungenau=0.5,
        weight_fraglich=0.0,
    ),
    AblationConfig(
        config_id="ablation_05_hn_strict_random_preselected",
        hard_negative_mode="strict",
        hard_negative_selection="random_preselected",
        include_fraglich_in_main_training=False,
        weight_eindeutig=1.0,
        weight_mehrdeutig=1.0,
        weight_ungenau=0.5,
        weight_fraglich=0.0,
    ),
]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuehrt eine 5er-Ablation auf Basis von Training/run_training_pipeline.py aus."
    )
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--expected-file", required=True)
    parser.add_argument("--base-model", default="BAAI/bge-m3")
    parser.add_argument("--hard-negatives-file", default="")
    parser.add_argument("--query-class-map-file", default="")
    parser.add_argument("--rule-policy-file", default="")
    parser.add_argument("--canary-file", default="")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-hard-negatives", type=int, default=1)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--max-per-positive", type=int, default=0)

    parser.add_argument(
        "--stop-before-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default: nur Pre-Train Gates/Manifest je Ablation, kein model.fit.",
    )
    parser.add_argument(
        "--qa-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sanitize-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--output-root", default="Training/outputs/ablation")
    parser.add_argument("--run-prefix", default="ablation")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def build_pipeline_command(
    *,
    args: argparse.Namespace,
    config: AblationConfig,
    run_id: str,
    output_root: Path,
) -> list[str]:
    pairs_out = output_root / "pairs" / f"{run_id}.jsonl"
    model_out = output_root / "models" / run_id

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--query-file",
        args.query_file,
        "--expected-file",
        args.expected_file,
        "--base-model",
        args.base_model,
        "--pairs-out",
        str(pairs_out),
        "--output-dir",
        str(model_out),
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
        "--hard-negative-mode",
        config.hard_negative_mode,
        "--hard-negative-selection",
        config.hard_negative_selection,
        "--num-hard-negatives",
        str(args.num_hard_negatives),
        "--weight-eindeutig",
        str(config.weight_eindeutig),
        "--weight-mehrdeutig",
        str(config.weight_mehrdeutig),
        "--weight-ungenau",
        str(config.weight_ungenau),
        "--weight-fraglich",
        str(config.weight_fraglich),
        "--run-id",
        run_id,
    ]

    if config.include_fraglich_in_main_training:
        command.append("--include-fraglich-in-main-training")
    else:
        command.append("--no-include-fraglich-in-main-training")

    if args.deduplicate:
        command.append("--deduplicate")
    if args.max_per_positive > 0:
        command.extend(["--max-per-positive", str(args.max_per_positive)])

    if args.hard_negatives_file.strip():
        command.extend(["--hard-negatives-file", args.hard_negatives_file])
    if args.query_class_map_file.strip():
        command.extend(["--query-class-map-file", args.query_class_map_file])
    if args.rule_policy_file.strip():
        command.extend(["--rule-policy-file", args.rule_policy_file])
    if args.canary_file.strip():
        command.extend(["--canary-file", args.canary_file])

    if args.stop_before_train:
        command.append("--stop-before-train")
    else:
        command.append("--no-stop-before-train")

    if args.qa_preflight:
        command.append("--qa-preflight")
    else:
        command.append("--no-qa-preflight")

    if args.sanitize_dataset:
        command.append("--sanitize-dataset")
    else:
        command.append("--no-sanitize-dataset")

    return command


def main() -> None:
    args = parse_args()

    if not PIPELINE_SCRIPT.is_file():
        raise FileNotFoundError(f"Pipeline-Skript nicht gefunden: {PIPELINE_SCRIPT}")

    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for config in DEFAULT_ABLATIONS:
        run_id = f"{args.run_prefix}_{config.config_id}"

        if config.hard_negative_mode == "strict" and not args.hard_negatives_file.strip():
            rows.append(
                {
                    "run_id": run_id,
                    "config_id": config.config_id,
                    "status": "SKIPPED",
                    "exit_code": "",
                    "duration_seconds": "0.0",
                    "reason": "strict mode requires --hard-negatives-file",
                }
            )
            continue

        command = build_pipeline_command(
            args=args,
            config=config,
            run_id=run_id,
            output_root=output_root,
        )

        print(f"\n> {' '.join(command)}")
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
        elapsed = time.perf_counter() - started

        status = "PASS" if completed.returncode == 0 else "FAIL"
        rows.append(
            {
                "run_id": run_id,
                "config_id": config.config_id,
                "status": status,
                "exit_code": str(completed.returncode),
                "duration_seconds": f"{elapsed:.6f}",
                "reason": "",
                "hard_negative_mode": config.hard_negative_mode,
                "hard_negative_selection": config.hard_negative_selection,
                "include_fraglich_in_main_training": str(config.include_fraglich_in_main_training),
                "weight_eindeutig": str(config.weight_eindeutig),
                "weight_mehrdeutig": str(config.weight_mehrdeutig),
                "weight_ungenau": str(config.weight_ungenau),
                "weight_fraglich": str(config.weight_fraglich),
            }
        )

        if completed.returncode != 0 and args.fail_fast:
            break

    summary_csv = output_root / "ablation_summary.csv"
    summary_json = output_root / "ablation_summary.json"

    fieldnames = [
        "run_id",
        "config_id",
        "status",
        "exit_code",
        "duration_seconds",
        "reason",
        "hard_negative_mode",
        "hard_negative_selection",
        "include_fraglich_in_main_training",
        "weight_eindeutig",
        "weight_mehrdeutig",
        "weight_ungenau",
        "weight_fraglich",
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "query_file": args.query_file,
        "expected_file": args.expected_file,
        "base_model": args.base_model,
        "stop_before_train": bool(args.stop_before_train),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nAblation summary CSV: {summary_csv}")
    print(f"Ablation summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
