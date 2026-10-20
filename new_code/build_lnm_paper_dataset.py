#!/usr/bin/env python3
"""
Build LNM task manifests per ThyroidXAgent paper layout:
  - ZJH-8K malignant primary-tumor images (local: Malignant_ultrasound_images_cropped)
  - LymphUs Center 1 train + Center 2 external test (local: Lymph_Node_Metastasis_fake)

Outputs JSON manifests under Classifaction_Data (no bulk image copy).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAPER_ZJH_MALIGNANT_IMAGES = 4756
PAPER_ZJH_VAL_CASES = 20
PAPER_ZJH_VAL_IMAGES = 183
PAPER_LYMPHUS_C1_TRAIN = 180
PAPER_LYMPHUS_C2_TEST = 158
PAPER_TRAIN_TOTAL = 4936  # 4756 + 180


def patient_from_zjh_filename(filename: str) -> str:
    filename = filename.replace("\\", "/")
    parts = filename.split("/")
    if len(parts) >= 2:
        return parts[-2]
    stem = Path(filename).stem
    match = re.match(r"^([^_]+)_", stem)
    return match.group(1) if match else stem


def load_zjh_labeled(zjh_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in ("train_labels.json", "test_labels.json"):
        path = zjh_root / name
        if not path.exists():
            raise FileNotFoundError(path)
        items = json.loads(path.read_text(encoding="utf-8"))
        for row in items:
            label = row.get("LNM_CN01")
            if label not in (0, 1):
                continue
            rel = str(row["filename"]).replace("\\", "/")
            if not (zjh_root / rel).exists():
                raise FileNotFoundError(f"missing image: {zjh_root / rel}")
            records.append(
                {
                    "filename": rel,
                    "LNM_CN01": int(label),
                    "patient_id": patient_from_zjh_filename(rel),
                    "source": "zjh8k_malignant_primary",
                    "image_root_key": "zjh",
                }
            )
    return records


def pick_zjh_relative_path(candidates: list[str], basename: str) -> str:
    stem = Path(basename).stem
    match = re.match(r"^([^_]+)_", stem)
    patient_hint = match.group(1) if match else stem
    for rel in candidates:
        if f"/{patient_hint}/" in rel or rel.startswith(f"{patient_hint}/"):
            return rel
    return sorted(candidates)[-1]


def build_zjh_basename_index(zjh_root: Path) -> dict[str, str]:
    buckets: dict[str, list[str]] = defaultdict(list)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for path in zjh_root.rglob("*"):
        if path.suffix.lower() not in exts:
            continue
        buckets[path.name].append(path.relative_to(zjh_root).as_posix())
    return {name: pick_zjh_relative_path(paths, name) for name, paths in buckets.items()}


def load_zjh_from_bm_labels(bm_path: Path, zjh_root: Path) -> list[dict[str, Any]]:
    items = json.loads(bm_path.read_text(encoding="utf-8"))
    index = build_zjh_basename_index(zjh_root)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in items:
        label = row.get("LNM_CN01")
        if label not in (0, 1):
            continue
        basename = str(row["filename"]).replace("\\", "/")
        rel = index.get(Path(basename).name)
        if rel is None:
            missing.append(basename)
            continue
        if not (zjh_root / rel).exists():
            missing.append(basename)
            continue
        records.append(
            {
                "filename": rel,
                "LNM_CN01": int(label),
                "patient_id": patient_from_zjh_filename(rel),
                "source": "zjh8k_malignant_primary",
                "image_root_key": "zjh",
                "zjh_bm_filename": Path(basename).name,
            }
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} ZJH-8K BM images not found under {zjh_root} (first: {missing[0]})"
        )
    return records


def select_val_patients(
    patients: dict[str, list[dict[str, Any]]],
    n_cases: int,
    target_images: int,
    seed: int,
    search_trials: int,
) -> set[str]:
    sizes = [(pid, len(rows), rows[0]["LNM_CN01"]) for pid, rows in patients.items()]
    if len(sizes) < n_cases:
        raise ValueError(f"need at least {n_cases} patients, got {len(sizes)}")

    rng = random.Random(seed)
    best_diff = 10**9
    best_subset: list[tuple[str, int, int]] | None = None

    for _ in range(search_trials):
        rng.shuffle(sizes)
        chosen = sizes[:n_cases]
        total = sum(x[1] for x in chosen)
        diff = abs(total - target_images)
        if diff < best_diff:
            best_diff = diff
            best_subset = chosen

    if best_subset is None:
        raise RuntimeError("failed to select validation patients")

    return {pid for pid, _, _ in best_subset}


def lymphus_records(
    labels_path: Path,
    center: str,
    split: str,
    image_root_key: str = "lymphus_fake",
) -> list[dict[str, Any]]:
    items = json.loads(labels_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    token = f"_{center}"
    for row in items:
        fn = str(row["filename"])
        if token not in fn:
            continue
        label = row.get("LNM_CN01")
        if label not in (0, 1):
            continue
        stem = Path(fn).stem
        patient_id = stem.split("_")[0] if "_" in stem else stem
        out.append(
            {
                "filename": fn,
                "LNM_CN01": int(label),
                "patient_id": patient_id,
                "source": f"lymphus_{center}",
                "image_root_key": image_root_key,
                "split": split,
            }
        )
    return out


def lymphus_records_from_labels(
    labels_path: Path,
    split: str,
    source: str,
    image_root_key: str,
) -> list[dict[str, Any]]:
    items = json.loads(labels_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for row in items:
        fn = str(row["filename"])
        label = row.get("LNM_CN01")
        if label not in (0, 1):
            continue
        stem = Path(fn).stem
        patient_id = stem.split("_")[0] if "_" in stem else stem
        out.append(
            {
                "filename": fn,
                "LNM_CN01": int(label),
                "patient_id": patient_id,
                "source": source,
                "image_root_key": image_root_key,
                "split": split,
            }
        )
    return out


def dump_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LNM paper-style dataset manifests")
    parser.add_argument(
        "--classification_data_root",
        type=Path,
        default=Path("/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data"),
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=("fake", "zjh8k_lymphus"),
        default="fake",
        help="fake: local cropped labels + Lymph_Node_Metastasis_fake; "
        "zjh8k_lymphus: ZJH-8K_BM_labels + real Lymph_Node_Metastasis C1/C2",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="defaults to LNM_ThyroidXAgent_ZJH_LymphUs_fake or LNM_ThyroidXAgent_ZJH8K_LymphUs",
    )
    parser.add_argument(
        "--zjh_bm_labels",
        type=Path,
        default=Path(
            "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ZJH-8K/ZJH-8K_BM_labels.json"
        ),
    )
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--val_search_trials", type=int, default=50_000)
    args = parser.parse_args()

    root = args.classification_data_root.resolve()
    zjh_root = root / "Malignant_ultrasound_images_cropped"
    out_dir = root / (
        args.output_name
        or (
            "LNM_ThyroidXAgent_ZJH8K_LymphUs"
            if args.profile == "zjh8k_lymphus"
            else "LNM_ThyroidXAgent_ZJH_LymphUs_fake"
        )
    )

    if args.profile == "zjh8k_lymphus":
        zjh_all = load_zjh_from_bm_labels(args.zjh_bm_labels.resolve(), zjh_root)
        lymphus_root = root / "Lymph_Node_Metastasis"
        lymphus_c1 = lymphus_records_from_labels(
            lymphus_root / "LymphUs_train_labels.json",
            split="train",
            source="lymphus_center1",
            image_root_key="lymphus_c1",
        )
        lymphus_c2 = lymphus_records_from_labels(
            lymphus_root / "LymphUs_test_labels.json",
            split="test",
            source="lymphus_center2",
            image_root_key="lymphus_c2",
        )
        image_roots = {
            "zjh": str(zjh_root),
            "lymphus_c1": str(lymphus_root / "center1/images"),
            "lymphus_c2": str(lymphus_root / "center2/images"),
        }
        zjh_label_source = "zjh8k_bm"
        lymphus_source = "real"
        lymphus_note = "LymphUs uses Lymph_Node_Metastasis (center1 train, center2 test)."
    else:
        lymphus_fake = root / "Lymph_Node_Metastasis_fake"
        zjh_all = load_zjh_labeled(zjh_root)
        lymphus_c1 = lymphus_records(
            lymphus_fake / "LymphUs_train_labels.json",
            center="center1",
            split="train",
            image_root_key="lymphus_fake",
        )
        lymphus_c2 = lymphus_records(
            lymphus_fake / "LymphUs_test_labels.json",
            center="center2",
            split="test",
            image_root_key="lymphus_fake",
        )
        image_roots = {
            "zjh": str(zjh_root),
            "lymphus_fake": str(lymphus_fake / "images"),
        }
        zjh_label_source = "malignant_cropped_train_test_json"
        lymphus_source = "fake"
        lymphus_note = "LymphUs uses Lymph_Node_Metastasis_fake; center1/center2 filtered by filename token."
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in zjh_all:
        by_patient[row["patient_id"]].append(row)

    val_patients = select_val_patients(
        by_patient,
        n_cases=PAPER_ZJH_VAL_CASES,
        target_images=PAPER_ZJH_VAL_IMAGES,
        seed=args.val_seed,
        search_trials=args.val_search_trials,
    )

    zjh_train: list[dict[str, Any]] = []
    zjh_val: list[dict[str, Any]] = []
    for pid, rows in by_patient.items():
        split = "val" if pid in val_patients else "train"
        for row in rows:
            entry = {**row, "split": split}
            if split == "val":
                zjh_val.append(entry)
            else:
                zjh_train.append(entry)

    if len(lymphus_c1) != PAPER_LYMPHUS_C1_TRAIN:
        raise ValueError(f"expected {PAPER_LYMPHUS_C1_TRAIN} LymphUs center1 train, got {len(lymphus_c1)}")
    if len(lymphus_c2) != PAPER_LYMPHUS_C2_TEST:
        raise ValueError(f"expected {PAPER_LYMPHUS_C2_TEST} LymphUs center2 test, got {len(lymphus_c2)}")

    train_combined = zjh_train + lymphus_c1
    for row in lymphus_c1:
        row["modality"] = "lymph_node_ultrasound"
    for row in zjh_train:
        row["modality"] = "primary_tumor_ultrasound"
    for row in zjh_val:
        row["modality"] = "primary_tumor_ultrasound"
    for row in lymphus_c2:
        row["modality"] = "lymph_node_ultrasound"

    cn0_cases = sum(1 for pid in by_patient if by_patient[pid][0]["LNM_CN01"] == 0)
    cn1_cases = sum(1 for pid in by_patient if by_patient[pid][0]["LNM_CN01"] == 1)
    cn0_img = sum(len(by_patient[p]) for p in by_patient if by_patient[p][0]["LNM_CN01"] == 0)
    cn1_img = sum(len(by_patient[p]) for p in by_patient if by_patient[p][0]["LNM_CN01"] == 1)

    metadata = {
        "description": "LNM task manifests aligned with ThyroidXAgent Methods 4.1 / Results 2.4",
        "paper_targets": {
            "zjh_malignant_images": PAPER_ZJH_MALIGNANT_IMAGES,
            "zjh_val_cases": PAPER_ZJH_VAL_CASES,
            "zjh_val_images": PAPER_ZJH_VAL_IMAGES,
            "lymphus_center1_train_images": PAPER_LYMPHUS_C1_TRAIN,
            "lymphus_center2_test_images": PAPER_LYMPHUS_C2_TEST,
            "combined_train_images": PAPER_TRAIN_TOTAL,
        },
        "actual_counts": {
            "zjh_labeled_images_total": len(zjh_all),
            "zjh_patients_total": len(by_patient),
            "zjh_cn0_cases": cn0_cases,
            "zjh_cn1_cases": cn1_cases,
            "zjh_cn0_images": cn0_img,
            "zjh_cn1_images": cn1_img,
            "zjh_train_images": len(zjh_train),
            "zjh_val_images": len(zjh_val),
            "zjh_val_cases": len(val_patients),
            "lymphus_center1_train": len(lymphus_c1),
            "lymphus_center2_test": len(lymphus_c2),
            "train_combined_images": len(train_combined),
        },
        "val_patient_ids": sorted(val_patients),
        "val_selection": {
            "seed": args.val_seed,
            "search_trials": args.val_search_trials,
            "target_val_images": PAPER_ZJH_VAL_IMAGES,
        },
        "image_roots": image_roots,
        "zjh_label_source": zjh_label_source,
        "zjh_bm_labels_path": str(args.zjh_bm_labels.resolve()) if args.profile == "zjh8k_lymphus" else None,
        "lymphus_source": lymphus_source,
        "notes": [
            "ZJH images: Malignant_ultrasound_images_cropped.",
            f"ZJH labels: {zjh_label_source}.",
            lymphus_note,
            "train_combined mixes image roots; use new_code/train.py with --image_roots_json.",
            "See actual_counts vs paper_targets in this file.",
        ],
    }

    manifest_dir = out_dir / "manifests"
    dump_json(image_roots, out_dir / "image_roots.json")
    dump_json(metadata, out_dir / "dataset_metadata.json")
    dump_json(zjh_all, manifest_dir / "zjh_all_labeled.json")
    dump_json(zjh_train, manifest_dir / "zjh_train.json")
    dump_json(zjh_val, manifest_dir / "zjh_val.json")
    dump_json(lymphus_c1, manifest_dir / "lymphus_center1_train.json")
    dump_json(lymphus_c2, manifest_dir / "lymphus_center2_test.json")
    dump_json(train_combined, manifest_dir / "train_combined_zjh_plus_lymphus_c1.json")

    def llmn_only_record(row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"filename": row["filename"], "LNM_CN01": row["LNM_CN01"]}
        for key in ("source", "image_root_key", "patient_id"):
            if key in row:
                out[key] = row[key]
        return out

    # new_code-friendly slices (LymphUs rows keep source/image_root_key for patient_id resolution)
    dump_json([llmn_only_record(r) for r in zjh_train], manifest_dir / "zjh_train_LLNM_only.json")
    dump_json([llmn_only_record(r) for r in zjh_val], manifest_dir / "zjh_val_LLNM_only.json")
    dump_json([llmn_only_record(r) for r in lymphus_c1], manifest_dir / "lymphus_center1_train_LLNM_only.json")
    dump_json([llmn_only_record(r) for r in lymphus_c2], manifest_dir / "lymphus_center2_test_LLNM_only.json")

    readme = f"""# LNM dataset (ThyroidXAgent paper layout)

Generated by `new_code/build_lnm_paper_dataset.py`.

## Image roots

| Key | Path |
|-----|------|
| zjh | `{image_roots['zjh']}` |
{chr(10).join(f"| {k} | `{v}` |" for k, v in image_roots.items() if k != "zjh")}

## Splits (paper intent)

| Manifest | Role |
|----------|------|
| `manifests/zjh_val.json` | ZJH validation ({metadata['actual_counts']['zjh_val_cases']} cases, {metadata['actual_counts']['zjh_val_images']} images; paper: 20 / 183) |
| `manifests/zjh_train.json` | ZJH training pool (excludes val patients) |
| `manifests/lymphus_center1_train.json` | LymphUs Center 1 ({PAPER_LYMPHUS_C1_TRAIN} images) |
| `manifests/lymphus_center2_test.json` | External test, LymphUs Center 2 ({PAPER_LYMPHUS_C2_TEST} images) |
| `manifests/train_combined_zjh_plus_lymphus_c1.json` | Paper training queue: ZJH train + LymphUs C1 |

## Counts vs paper

- Paper combined train: **{PAPER_TRAIN_TOTAL}** (= {PAPER_ZJH_MALIGNANT_IMAGES} ZJH + {PAPER_LYMPHUS_C1_TRAIN} LymphUs C1)
- This build combined train: **{metadata['actual_counts']['train_combined_images']}** (see `dataset_metadata.json`)

## Training with LLNM-Net `new_code/train.py`

Single-root examples:

```bash
# ZJH train / val
python new_code/train.py \\
  --train_json {manifest_dir}/zjh_train_LLNM_only.json \\
  --test_json {manifest_dir}/zjh_val_LLNM_only.json \\
  --image_root {image_roots['zjh']} \\
  --output_dir ./new_code/runs/LNM_paper_zjh

# LymphUs C1 train + C2 external (multi-root manifests)
python new_code/train.py \\
  --train_json {manifest_dir}/lymphus_center1_train.json \\
  --test_json {manifest_dir}/lymphus_center2_test.json \\
  --image_roots_json {out_dir}/image_roots.json \\
  --output_dir ./new_code/runs/LNM_paper_lymphus

# B9 merged queue (ZJH train + C1) → C2 external
python new_code/train.py \\
  --train_json {manifest_dir}/train_combined_zjh_plus_lymphus_c1.json \\
  --test_json {manifest_dir}/lymphus_center2_test.json \\
  --image_roots_json {out_dir}/image_roots.json \\
  --output_dir ./new_code/runs/b9_paper_full_train
```
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(metadata["actual_counts"], ensure_ascii=False, indent=2))
    print(f"wrote dataset to {out_dir}")


if __name__ == "__main__":
    main()
