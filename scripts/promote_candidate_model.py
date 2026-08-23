"""Gate candidate ModernBERT artifacts and promote them to final."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_CANDIDATE = ROOT / "models" / "candidate" / "component_classifier" / "artifact"
RELATION_CANDIDATE = ROOT / "models" / "candidate" / "relation_classifier" / "artifact"
COMPONENT_FINAL = ROOT / "models" / "component_classifier" / "final"
RELATION_FINAL = ROOT / "models" / "relation_classifier" / "final"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"
HOLDOUT_PATH = ROOT / "eval_holdout_scenarios_v2.json"
# Primary, honest gate: macro-F1 on the locked, independently-generated gold holdout.
# This is what "real performance" means. Overridable via env for iterative work.
HOLDOUT_F1_THRESHOLD = float(os.getenv("GOLD_F1_THRESHOLD", "0.80"))
# Overfitting guard: a small train-vs-test macro-F1 gap proves the model is not
# memorizing. Promotion is blocked if the gap is too large even when F1 is high.
MAX_OVERFIT_GAP = float(os.getenv("MAX_OVERFIT_GAP", "0.15"))
# Auxiliary hand-written hard cases stay informational (recorded, soft floor only).
HARD_CASE_ACCURACY_THRESHOLD = float(os.getenv("HARD_CASE_FLOOR", "0.75"))


def assert_artifact(path: Path) -> None:
    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise SystemExit(f"Missing candidate files in {path}: {missing}")
    model_file = path / "model.safetensors"
    if model_file.stat().st_size < 100_000_000:
        raise SystemExit(f"Candidate model is too small: {model_file}")
    with model_file.open("rb") as fh:
        if b"git-lfs.github.com/spec" in fh.read(256):
            raise SystemExit(f"Candidate model is an LFS pointer: {model_file}")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_json(command: list[str]) -> dict:
    try:
        output = subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.output
    return json.loads(output)


def copy_tree(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)


def main() -> None:
    assert_artifact(COMPONENT_CANDIDATE)
    assert_artifact(RELATION_CANDIDATE)
    if not HOLDOUT_PATH.exists():
        raise SystemExit(
            "Missing eval_holdout_scenarios_v2.json. Generate a fresh OpenRouter "
            "holdout set before promotion."
        )
    hard = run_json(
        [
            sys.executable,
            "scripts/evaluate_hard_model_cases.py",
            "--component-model-dir",
            str(COMPONENT_CANDIDATE),
            "--relation-model-dir",
            str(RELATION_CANDIDATE),
        ]
    )
    holdout = run_json(
        [
            sys.executable,
            "scripts/evaluate_real_scenarios.py",
            "--component-model-dir",
            str(COMPONENT_CANDIDATE),
            "--relation-model-dir",
            str(RELATION_CANDIDATE),
            "--load-from-file",
            str(HOLDOUT_PATH),
            "--json",
            "--enforce-thresholds",
        ]
    )
    debate = run_json(
        [
            sys.executable,
            "scripts/evaluate_debate_acceptance.py",
            "--component-model-dir",
            str(COMPONENT_CANDIDATE),
            "--relation-model-dir",
            str(RELATION_CANDIDATE),
        ]
    )
    # --- Overfitting guard (train vs test gap from the training summaries) --- #
    component_training = read_json(ROOT / "models" / "candidate" / "component_classifier" / "training_summary.json")
    relation_training = read_json(ROOT / "models" / "candidate" / "relation_classifier" / "training_summary.json")
    comp_gap = float(component_training.get("overfitting_gap", 0.0))
    rel_gap = float(relation_training.get("overfitting_gap", 0.0))

    failures: list[str] = []
    # PRIMARY gate: real performance on the locked gold holdout.
    if holdout["component_macro_f1"] < HOLDOUT_F1_THRESHOLD:
        failures.append(
            f"gold-holdout component macro-F1 {holdout['component_macro_f1']:.4f} < {HOLDOUT_F1_THRESHOLD:.2f}"
        )
    if holdout["relation_macro_f1"] < HOLDOUT_F1_THRESHOLD:
        failures.append(
            f"gold-holdout relation macro-F1 {holdout['relation_macro_f1']:.4f} < {HOLDOUT_F1_THRESHOLD:.2f}"
        )
    # OVERFITTING gate: train-test gap must stay small.
    if comp_gap > MAX_OVERFIT_GAP:
        failures.append(f"component overfitting gap {comp_gap:.4f} > {MAX_OVERFIT_GAP:.2f}")
    if rel_gap > MAX_OVERFIT_GAP:
        failures.append(f"relation overfitting gap {rel_gap:.4f} > {MAX_OVERFIT_GAP:.2f}")
    # SOFT floor: auxiliary hard-cases are informational but block egregious regressions.
    if hard["accuracy"] < HARD_CASE_ACCURACY_THRESHOLD:
        failures.append(
            f"hard-case accuracy {hard['accuracy']:.4f} < soft floor {HARD_CASE_ACCURACY_THRESHOLD:.2f}"
        )
    if failures:
        raise SystemExit("Promotion blocked:\n  - " + "\n  - ".join(failures))

    copy_tree(COMPONENT_CANDIDATE, COMPONENT_FINAL)
    copy_tree(RELATION_CANDIDATE, RELATION_FINAL)
    relation_calibration = read_json(RELATION_CANDIDATE / "calibration.json")
    if not relation_calibration:
        raise SystemExit("Missing relation calibration metadata")

    def gen_block(summary: dict) -> dict:
        """train/val/test macro-F1 side by side + overfitting gap (honest view)."""
        return {
            "train_f1": summary.get("train", {}).get("train_f1"),
            "val_f1": summary.get("eval", {}).get("eval_f1"),
            "test_f1": summary.get("test", {}).get("test_f1"),
            "overfitting_gap": summary.get("overfitting_gap"),
        }

    metrics = {
        "model": {
            "primary_metric": "gold_holdout_macro_f1",
            "gold_holdout": {
                "dataset": str(HOLDOUT_PATH.relative_to(ROOT)),
                "component_macro_f1": holdout["component_macro_f1"],
                "relation_macro_f1": holdout["relation_macro_f1"],
                "component_total": holdout["component_total"],
                "relation_total": holdout["relation_total"],
                "threshold": HOLDOUT_F1_THRESHOLD,
            },
            "generalization": {
                "component": gen_block(component_training),
                "relation": gen_block(relation_training),
                "max_overfit_gap": MAX_OVERFIT_GAP,
            },
            "auxiliary_hard_cases": {
                "component_accuracy": hard["component_accuracy"],
                "relation_accuracy": hard["relation_accuracy"],
                "claim_recall": hard["claim_recall"],
                "none_precision": hard["none_precision"],
                "note": "informational only; not the promotion gate",
            },
            "debate_acceptance": [
                {
                    "name": item["name"],
                    "component_accuracy": item["component_accuracy"],
                    "relation_accuracy": item["relation_accuracy"],
                }
                for item in debate["debates"]
            ],
            "component_training": {
                "dataset_version": component_training.get("dataset_version"),
                "dataset_counts": component_training.get("dataset_counts"),
                "class_weights": component_training.get("class_weights"),
                "eval": component_training.get("eval"),
                "test": component_training.get("test"),
            },
            "relation_training": {
                "dataset_version": relation_training.get("dataset_version"),
                "dataset_counts": relation_training.get("dataset_counts"),
                "class_weights": relation_training.get("class_weights"),
                "calibration": relation_calibration,
                "eval": relation_training.get("eval"),
                "test": relation_training.get("test"),
            },
            "source": "scripts/promote_candidate_model.py",
        }
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"promoted": True, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
