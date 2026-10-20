#!/usr/bin/env python3
"""
Patient-level stratified K-fold CV on the B9 LNM training queue (ZJH train + LymphUs C1).

See docs/B9_LNM_构建计划.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from models.modeling_LLNM_Net import CONFIGS, LLNM_Net
from new_code.dataset import LLNMDataset
from new_code.engine import evaluate, train_one_epoch
from new_code.train import build_dataloader, load_weights
from new_code.utils import (
    apply_normalization,
    build_sample_records,
    compute_normalization_stats,
    dump_json,
    ensure_dir,
    load_image_roots_json,
    save_pickle,
    summarize_binary_predictions,
)


def parse_args() -> argparse.Namespace:
    default_data = Path(
        "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/LNM_ThyroidXAgent_ZJH_LymphUs_fake"
    )
    parser = argparse.ArgumentParser(description="B9 patient-level K-fold CV for LNM")
    parser.add_argument(
        "--train_manifest",
        type=str,
        default=str(default_data / "manifests/train_combined_zjh_plus_lymphus_c1.json"),
    )
    parser.add_argument(
        "--image_roots_json",
        type=str,
        default=str(default_data / "image_roots.json"),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--cv_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--radiomics_csv", type=str, default=None)
    parser.add_argument("--patient_info", type=str, default=None)
    parser.add_argument("--save_fold_models", action="store_true")
    parser.add_argument(
        "--cluster_key",
        type=str,
        choices=["patient_id", "filename"],
        default="patient_id",
        help="CV grouping key; use filename for LymphUs-only smoke tests if needed",
    )
    return parser.parse_args()


def cluster_id(sample: dict, cluster_key: str) -> str:
    if cluster_key == "filename":
        return str(sample["filename"])
    return str(sample["patient_id"])


def cluster_labels(samples: list[dict], cluster_key: str) -> dict[str, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        grouped[cluster_id(sample, cluster_key)].append(int(sample["LNM_CN01"]))
    labels: dict[str, int] = {}
    for cid, values in grouped.items():
        if len(set(values)) > 1:
            raise ValueError(f"cluster {cid} has mixed labels: {set(values)}")
        labels[cid] = values[0]
    return labels


def train_fold(
    train_samples: list[dict],
    val_samples: list[dict],
    args: argparse.Namespace,
    fold_dir: Path,
    device: torch.device,
) -> tuple[dict, dict]:
    norm_stats = compute_normalization_stats(train_samples)
    save_pickle(norm_stats, fold_dir / "train_norm_stats.pkl")

    train_dataset = LLNMDataset(
        samples=[apply_normalization(s, norm_stats) for s in train_samples],
        norm_stats=None,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )
    val_dataset = LLNMDataset(
        samples=[apply_normalization(s, norm_stats) for s in val_samples],
        norm_stats=None,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    config = CONFIGS["LLNM_Net"]
    model = LLNM_Net(config, args.image_size, zero_head=True, num_classes=1)
    if args.pretrained_weights:
        model = load_weights(model, args.pretrained_weights)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = BCEWithLogitsLoss()

    best_auroc = float("-inf")
    best_path = fold_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        val_metrics, val_outputs = evaluate(model, val_loader, device)
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), best_path)
        print(f"fold {fold_dir.name} epoch {epoch}/{args.epochs} val_auroc={val_metrics['auroc']:.4f}")

    state_dict = torch.load(best_path, map_location=device)
    model.load_state_dict(state_dict)
    val_metrics, val_outputs = evaluate(model, val_loader, device)
    if args.save_fold_models:
        torch.save(model.state_dict(), fold_dir / "final_best_model.pth")

    return val_metrics, val_outputs


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT_DIR / "new_code" / "runs" / f"b9_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    ensure_dir(output_dir)

    image_roots = load_image_roots_json(args.image_roots_json)
    default_root = image_roots.get("zjh", next(iter(image_roots.values())))
    samples = build_sample_records(
        manifest_path=args.train_manifest,
        image_root=default_root,
        image_roots=image_roots,
        patient_info_file=args.patient_info,
        radiomics_csv=args.radiomics_csv,
    )

    cid_to_label = cluster_labels(samples, args.cluster_key)
    cluster_ids = sorted(cid_to_label.keys())
    y = np.array([cid_to_label[cid] for cid in cluster_ids], dtype=int)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.cv_seed)
    fold_summaries: list[dict] = []
    oof_labels: list[int] = []
    oof_probs: list[float] = []
    oof_patient_ids: list[str] = []
    oof_paths: list[str] = []

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    dump_json(
        {
            "train_manifest": args.train_manifest,
            "image_roots_json": args.image_roots_json,
            "n_samples": len(samples),
            "n_clusters": len(cluster_ids),
            "cluster_key": args.cluster_key,
            "n_splits": args.n_splits,
            "cv_seed": args.cv_seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        output_dir / "cv_config.json",
    )

    for fold_idx, (train_cid_idx, val_cid_idx) in enumerate(skf.split(cluster_ids, y)):
        train_cids = {cluster_ids[i] for i in train_cid_idx}
        val_cids = {cluster_ids[i] for i in val_cid_idx}
        train_samples = [s for s in samples if cluster_id(s, args.cluster_key) in train_cids]
        val_samples = [s for s in samples if cluster_id(s, args.cluster_key) in val_cids]
        fold_dir = output_dir / f"fold_{fold_idx:02d}"
        ensure_dir(fold_dir)

        print(
            f"=== fold {fold_idx} | train {len(train_samples)} images / {len(train_cids)} clusters | "
            f"val {len(val_samples)} images / {len(val_cids)} clusters ==="
        )

        val_metrics, val_outputs = train_fold(train_samples, val_samples, args, fold_dir, device)
        fold_record = {
            "fold": fold_idx,
            "train_images": len(train_samples),
            "val_images": len(val_samples),
            "train_clusters": len(train_cids),
            "val_clusters": len(val_cids),
            "val_metrics_image": val_metrics,
            "val_metrics_patient": val_outputs["metrics_patient"],
        }
        fold_summaries.append(fold_record)
        dump_json(fold_record, fold_dir / "fold_summary.json")

        oof_labels.extend(val_outputs["labels"])
        oof_probs.extend(val_outputs["probs"])
        oof_patient_ids.extend(val_outputs["patient_ids"])
        oof_paths.extend(val_outputs["paths"])

    oof_summary = summarize_binary_predictions(
        np.asarray(oof_labels, dtype=int),
        np.asarray(oof_probs, dtype=float),
        oof_patient_ids,
    )
    cv_summary = {
        "folds": fold_summaries,
        "oof_metrics_image": oof_summary["metrics_image"],
        "oof_metrics_patient": oof_summary["metrics_patient"],
        "oof_n_images": len(oof_labels),
        "oof_n_patients": oof_summary["metrics_patient"]["n_patients"],
    }
    dump_json(cv_summary, output_dir / "cv_summary.json")
    dump_json(
        {
            "paths": oof_paths,
            "labels": oof_labels,
            "probs": oof_probs,
            "patient_ids": oof_patient_ids,
            "patient_predictions": oof_summary["patient_predictions"],
        },
        output_dir / "oof_predictions.json",
    )
    print(json.dumps(cv_summary["oof_metrics_patient"], ensure_ascii=False, indent=2))
    print(f"saved CV results to {output_dir}")


if __name__ == "__main__":
    main()
