"""Validate that production ModernBERT artifacts are present.

The large model files are stored with Git LFS. CI/CD should run this before
building the Docker image so a build never silently bakes LFS pointer files into
the container.
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = [
    Path("models/component_classifier/final/config.json"),
    Path("models/component_classifier/final/model.safetensors"),
    Path("models/component_classifier/final/tokenizer.json"),
    Path("models/component_classifier/final/tokenizer_config.json"),
    Path("models/relation_classifier/final/config.json"),
    Path("models/relation_classifier/final/model.safetensors"),
    Path("models/relation_classifier/final/tokenizer.json"),
    Path("models/relation_classifier/final/tokenizer_config.json"),
]

MIN_MODEL_BYTES = 100 * 1024 * 1024


def is_lfs_pointer(path: Path) -> bool:
    """Return True when a file is a Git LFS pointer instead of real content."""
    try:
        head = path.read_bytes()[:128]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def main() -> None:
    """Fail with a clear message if model files are missing or unresolved."""
    failures: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            failures.append(f"missing: {path}")
            continue
        if is_lfs_pointer(path):
            failures.append(f"unresolved Git LFS pointer: {path}")
            continue
        if path.name == "model.safetensors" and path.stat().st_size < MIN_MODEL_BYTES:
            failures.append(f"model file is unexpectedly small: {path}")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(
            "ModernBERT model artifacts are not ready for Docker build.\n"
            "Run `git lfs install && git lfs pull` and retry.\n"
            f"{details}"
        )

    print("ModernBERT model artifacts are present.")


if __name__ == "__main__":
    main()
