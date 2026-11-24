"""Translate workstation-specific manifest paths into deployed dataset paths.

Only existing files inside ROOT/dataset are accepted. The private Hugging Face
manifest is never modified; this normalizes the downloaded working copy.
"""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
import pandas as pd

PATH_COLUMNS = (
    "hysteresis_path", "zfc_fc_path", "manual_label_path",
    "ensemble_definition_path", "image_path",
)


def repair_manifest(root: Path) -> dict[str, object]:
    dataset_root = (root / "dataset").resolve()
    manifest = dataset_root / "ml_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError("Missing dataset/ml_manifest.csv in the private repository")
    df = pd.read_csv(manifest, dtype={"sample_id": "string"})
    required = {"split", "label", "hysteresis_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Manifest missing columns: " + ", ".join(sorted(missing)))

    by_filename: dict[str, list[Path]] = {}
    for p in dataset_root.rglob("*"):
        if p.is_file():
            by_filename.setdefault(p.name, []).append(p)

    repaired: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    for column in PATH_COLUMNS:
        if column not in df.columns:
            continue
        resolved_values = []
        for original in df[column]:
            if pd.isna(original) or not str(original).strip():
                resolved_values.append("")
                continue
            raw = str(original).strip()
            normalized = raw.replace("\\", "/")
            # PureWindowsPath handles drive letters independently of the host OS.
            filename = PureWindowsPath(raw).name
            parts = normalized.split("/")
            candidates: list[Path] = []
            # Preserve dataset-relative and repository-relative paths when supplied.
            if not PureWindowsPath(raw).drive and not normalized.startswith("/"):
                candidates.extend((root / normalized, dataset_root / normalized))
            if "dataset" in parts:
                pos = len(parts) - 1 - parts[::-1].index("dataset")
                candidates.append(root.joinpath(*parts[pos:]))
            candidates.extend(by_filename.get(filename, []) if len(by_filename.get(filename, [])) == 1 else [])
            hit = next(
                (p.resolve() for p in candidates if p.is_file() and p.resolve().is_relative_to(dataset_root)),
                None,
            )
            if hit is None:
                missing_counts[column] = missing_counts.get(column, 0) + 1
                # Never leave the original Windows workstation path in the deployed copy.
                resolved_values.append("")
            else:
                repaired[column] = repaired.get(column, 0) + 1
                resolved_values.append(str(hit))
        df[column] = resolved_values
    df.to_csv(manifest, index=False)
    return {"rows": len(df), "repaired": repaired, "unresolved": missing_counts}
