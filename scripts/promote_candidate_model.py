"""Gate candidate ModernBERT artifacts and promote them to final."""

from __future__ import annotations

import json
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
HOLDOUT_PATH = ROOT / "eval_holdout_scenarios.json"


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
            "Missing eval_holdout_scenarios.json. Generate a fresh OpenRouter "
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
    gate_errors = []
    if hard["component_accuracy"] < 0.75 or hard["relation_accuracy"] < 0.60:
        gate_errors.append("hard-case accuracy thresholds not met")
    if hard["claim_recall"] < 0.70 or hard["none_precision"] < 0.88:
        gate_errors.append("hard-case recall/precision thresholds not met")
    for item in debate["debates"]:
        if item["component_accuracy"] < 0.70 or item["relation_accuracy"] < 0.65:
            gate_errors.append(f"debate acceptance threshold not met: {item['name']}")
    if holdout["component_macro_f1"] < 0.80:
        gate_errors.append(
            f"holdout component macro-F1 below 0.80: {holdout['component_macro_f1']:.4f}"
        )
    if holdout["relation_macro_f1"] < 0.80:
        gate_errors.append(
            f"holdout relation macro-F1 below 0.80: {holdout['relation_macro_f1']:.4f}"
        )
    if gate_errors:
        raise SystemExit("Promotion blocked:\n- " + "\n- ".join(gate_errors))

    copy_tree(COMPONENT_CANDIDATE, COMPONENT_FINAL)
    copy_tree(RELATION_CANDIDATE, RELATION_FINAL)
    component_training = read_json(ROOT / "models" / "candidate" / "component_classifier" / "training_summary.json")
    relation_training = read_json(ROOT / "models" / "candidate" / "relation_classifier" / "training_summary.json")
    relation_calibration = read_json(RELATION_CANDIDATE / "calibration.json")
    if not relation_calibration:
        raise SystemExit("Missing relation calibration metadata")
    metrics = {
        "model": {
            "f1_components": hard["component_accuracy"],
            "f1_relations": hard["relation_accuracy"],
            "component_hard_case_accuracy": hard["component_accuracy"],
            "relation_hard_case_accuracy": hard["relation_accuracy"],
            "claim_recall": hard["claim_recall"],
            "none_precision": hard["none_precision"],
            "holdout": {
                "dataset": str(HOLDOUT_PATH.relative_to(ROOT)),
                "component_macro_f1": holdout["component_macro_f1"],
                "relation_macro_f1": holdout["relation_macro_f1"],
                "component_total": holdout["component_total"],
                "relation_total": holdout["relation_total"],
                "passed": holdout["passed"],
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
