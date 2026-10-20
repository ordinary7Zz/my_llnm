#!/usr/bin/env python3
"""
Pack a self-contained LNM B9 dataset directory:
  - ZJH: 4756 images with LNM_CN01 in {0,1} from ZJH-8K_BM_labels.json
  - LymphUs Center1 (180): rows in Lymph_Node_Metastasis_fake/LymphUs_train_labels.json with center1
  - LymphUs Center2 (158): Lymph_Node_Metastasis_fake/LymphUs_test_labels.json (center2)

Images are copied (not symlinked). Manifests match ThyroidXAgent naming (C1/C2 counts).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_lnm_paper_dataset import (
    PAPER_LYMPHUS_C1_TRAIN,
    PAPER_LYMPHUS_C2_TEST,
    PAPER_ZJH_MALIGNANT_IMAGES,
    PAPER_ZJH_VAL_CASES,
    PAPER_ZJH_VAL_IMAGES,
    dump_json,
    load_zjh_from_bm_labels,
    lymphus_records,
    select_val_patients,
)
from collections import defaultdict


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack ZJH8K + LymphUs C1/C2 images into one dataset dir")
    parser.add_argument(
        "--classification_data_root",
        type=Path,
        default=Path("/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data"),
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="LNM_ZJH8K_LymphUs_C1C2_packed",
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
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = args.classification_data_root.resolve()
    zjh_src_root = root / "Malignant_ultrasound_images_cropped"
    lymphus_fake = root / "Lymph_Node_Metastasis_fake"
    out_dir = root / args.output_name

    zjh_img_out = out_dir / "images" / "zjh"
    c1_img_out = out_dir / "images" / "lymphus_center1"
    c2_img_out = out_dir / "images" / "lymphus_center2"

    zjh_records = load_zjh_from_bm_labels(args.zjh_bm_labels.resolve(), zjh_src_root)
    if len(zjh_records) != PAPER_ZJH_MALIGNANT_IMAGES:
        raise ValueError(f"expected {PAPER_ZJH_MALIGNANT_IMAGES} ZJH images, got {len(zjh_records)}")

    lymphus_c1 = lymphus_records(
        lymphus_fake / "LymphUs_train_labels.json",
        center="center1",
        split="train",
        image_root_key="lymphus_c1",
    )
    lymphus_c2 = lymphus_records(
        lymphus_fake / "LymphUs_test_labels.json",
        center="center2",
        split="test",
        image_root_key="lymphus_c2",
    )
    if len(lymphus_c1) != PAPER_LYMPHUS_C1_TRAIN:
        raise ValueError(f"expected C1 {PAPER_LYMPHUS_C1_TRAIN}, got {len(lymphus_c1)}")
    if len(lymphus_c2) != PAPER_LYMPHUS_C2_TEST:
        raise ValueError(f"expected C2 {PAPER_LYMPHUS_C2_TEST}, got {len(lymphus_c2)}")

    fake_images = lymphus_fake / "images"
    print(f"copying {len(zjh_records)} ZJH images -> {zjh_img_out}")
    for row in zjh_records:
        rel = row["filename"]
        copy_file(zjh_src_root / rel, zjh_img_out / rel, args.dry_run)

    print(f"copying {len(lymphus_c1)} Center1 images -> {c1_img_out}")
    for row in lymphus_c1:
        fn = row["filename"]
        copy_file(fake_images / fn, c1_img_out / fn, args.dry_run)

    print(f"copying {len(lymphus_c2)} Center2 images -> {c2_img_out}")
    for row in lymphus_c2:
        fn = row["filename"]
        copy_file(fake_images / fn, c2_img_out / fn, args.dry_run)

    by_patient: dict[str, list] = defaultdict(list)
    for row in zjh_records:
        by_patient[row["patient_id"]].append(row)

    val_patients = select_val_patients(
        by_patient,
        n_cases=PAPER_ZJH_VAL_CASES,
        target_images=PAPER_ZJH_VAL_IMAGES,
        seed=args.val_seed,
        search_trials=args.val_search_trials,
    )

    zjh_train, zjh_val = [], []
    for pid, rows in by_patient.items():
        split = "val" if pid in val_patients else "train"
        for row in rows:
            entry = {**row, "split": split, "modality": "primary_tumor_ultrasound"}
            (zjh_val if split == "val" else zjh_train).append(entry)

    for row in lymphus_c1:
        row["modality"] = "lymph_node_ultrasound"
    for row in lymphus_c2:
        row["modality"] = "lymph_node_ultrasound"

    train_combined = zjh_train + lymphus_c1

    image_roots = {
        "zjh": str(zjh_img_out),
        "lymphus_c1": str(c1_img_out),
        "lymphus_c2": str(c2_img_out),
    }

    metadata = {
        "description": "Self-contained LNM B9 pack: ZJH-8K BM (4756) + LymphUs C1/C2",
        "paper_splits": {
            "zjh_malignant_lnm_labeled": PAPER_ZJH_MALIGNANT_IMAGES,
            "lymphus_center1_train": PAPER_LYMPHUS_C1_TRAIN,
            "lymphus_center2_external_test": PAPER_LYMPHUS_C2_TEST,
        },
        "actual_counts": {
            "zjh_copied": len(zjh_records),
            "zjh_train_images": len(zjh_train),
            "zjh_val_images": len(zjh_val),
            "zjh_val_cases": len(val_patients),
            "lymphus_center1": len(lymphus_c1),
            "lymphus_center2": len(lymphus_c2),
            "train_combined_images": len(train_combined),
        },
        "val_patient_ids": sorted(val_patients),
        "zjh_label_source": str(args.zjh_bm_labels.resolve()),
        "zjh_image_source": str(zjh_src_root),
        "lymphus_label_source": {
            "center1": str(lymphus_fake / "LymphUs_train_labels.json"),
            "center2": str(lymphus_fake / "LymphUs_test_labels.json"),
        },
        "lymphus_image_source": str(fake_images),
        "notes": [
            "LymphUs files are copied from Lymph_Node_Metastasis_fake/images.",
            "Center1 = 180 images from train_labels with 'center1' in filename; "
            "Center2 = 158 from test_labels (center2), matching ThyroidXAgent C1/C2 split.",
            "ZJH paths under images/zjh/ preserve Malignant_ultrasound_images_cropped layout.",
        ],
    }

    manifest_dir = out_dir / "manifests"
    dump_json(image_roots, out_dir / "image_roots.json")
    dump_json(metadata, out_dir / "dataset_metadata.json")
    dump_json(zjh_records, manifest_dir / "zjh_all_labeled.json")
    dump_json(zjh_train, manifest_dir / "zjh_train.json")
    dump_json(zjh_val, manifest_dir / "zjh_val.json")
    dump_json(lymphus_c1, manifest_dir / "lymphus_center1_train.json")
    dump_json(lymphus_c2, manifest_dir / "lymphus_center2_test.json")
    dump_json(train_combined, manifest_dir / "train_combined_zjh_plus_lymphus_c1.json")

    readme = f"""# LNM packed dataset (ZJH-8K 4756 + LymphUs C1/C2)

Built by `new_code/pack_lnm_b9_dataset.py`.

## Splits (ThyroidXAgent wording)

| Split | Count | On-disk folder |
|-------|------:|----------------|
| ZJH malignant with LNM label | 4756 | `images/zjh/` |
| **LymphUs Center1** (training supplement) | **180** | `images/lymphus_center1/` |
| **LymphUs Center2** (external test) | **158** | `images/lymphus_center2/` |
| Combined training queue (ZJH train + C1) | {len(train_combined)} | manifests |

LymphUs pixels are stored under `lymphus_center1` / `lymphus_center2`; labels are taken from
`Lymph_Node_Metastasis_fake/LymphUs_train_labels.json` (C1) and `LymphUs_test_labels.json` (C2).

## Training

```bash
DATA={out_dir}
python new_code/run_b9_patient_cv.py \\
  --train_manifest $DATA/manifests/train_combined_zjh_plus_lymphus_c1.json \\
  --image_roots_json $DATA/image_roots.json \\
  --output_dir ./new_code/runs/b9_packed_cv
```
"""
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(metadata["actual_counts"], ensure_ascii=False, indent=2))
    print(f"done -> {out_dir}" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
