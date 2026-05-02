from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from models.modeling_LLNM_Net import CONFIGS, LLNM_Net
from new_code.dataset import LLNMDataset
from new_code.engine import evaluate, train_one_epoch
from new_code.utils import compute_normalization_stats, dump_json, ensure_dir, load_pickle, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLNM-Net from JSON manifests")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--patient_info", type=str, default=None)
    parser.add_argument("--radiomics_csv", type=str, default=None)
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--default_report", type=str, default="")
    parser.add_argument("--default_age", type=float, default=50.0)
    parser.add_argument("--default_sex", type=float, default=1.0)
    parser.add_argument("--shape_feature", type=str, default="original_shape2D_Elongation")
    parser.add_argument("--echo_feature", type=str, default="original_firstorder_Mean")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_best_metric", type=str, choices=["auroc", "loss"], default="auroc")
    parser.add_argument("--save_all_epochs", action="store_true")
    return parser.parse_args()


def load_weights(model: torch.nn.Module, weight_path: str) -> torch.nn.Module:
    pretrained_weights = torch.load(weight_path, map_location=torch.device("cpu"))
    model_weights = model.state_dict()
    if any(k.startswith("module.") for k in pretrained_weights.keys()):
        pretrained_weights = {k.replace("module.", ""): v for k, v in pretrained_weights.items()}
    model_weights.update({k: v for k, v in pretrained_weights.items() if k in model_weights})
    model.load_state_dict(model_weights)
    return model


def build_dataloader(dataset: LLNMDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("new_code") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(output_dir)

    train_dataset_raw = LLNMDataset(
        manifest_path=args.train_json,
        image_root=args.image_root,
        patient_info_file=args.patient_info,
        radiomics_csv=args.radiomics_csv,
        default_report=args.default_report,
        default_age=args.default_age,
        default_sex=args.default_sex,
        shape_feature=args.shape_feature,
        echo_feature=args.echo_feature,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )
    norm_stats = compute_normalization_stats(train_dataset_raw.samples)
    save_pickle(norm_stats, output_dir / "train_norm_stats.pkl")

    train_dataset = LLNMDataset(
        manifest_path=args.train_json,
        image_root=args.image_root,
        patient_info_file=args.patient_info,
        radiomics_csv=args.radiomics_csv,
        default_report=args.default_report,
        default_age=args.default_age,
        default_sex=args.default_sex,
        shape_feature=args.shape_feature,
        echo_feature=args.echo_feature,
        norm_stats=norm_stats,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True)

    test_dataset = None
    test_loader = None
    if args.test_json:
        test_dataset = LLNMDataset(
            manifest_path=args.test_json,
            image_root=args.image_root,
            patient_info_file=args.patient_info,
            radiomics_csv=args.radiomics_csv,
            default_report=args.default_report,
            default_age=args.default_age,
            default_sex=args.default_sex,
            shape_feature=args.shape_feature,
            echo_feature=args.echo_feature,
            norm_stats=norm_stats,
            image_size=args.image_size,
            resize_size=args.resize_size,
        )
        test_loader = build_dataloader(test_dataset, args.batch_size, args.num_workers, shuffle=False)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    config = CONFIGS["LLNM_Net"]
    model = LLNM_Net(config, args.image_size, zero_head=True, num_classes=1)
    if args.pretrained_weights:
        model = load_weights(model, args.pretrained_weights)
    model = model.to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1 and requested_device == "cuda":
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = BCEWithLogitsLoss()

    history: list[dict] = []
    best_score = float("-inf") if args.save_best_metric == "auroc" else float("inf")
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss_fn=loss_fn,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
        }

        print(
            f"epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_auroc={train_metrics['auroc']:.4f}"
        )

        candidate_score = train_metrics["auroc"] if args.save_best_metric == "auroc" else train_loss

        if test_loader is not None:
            test_metrics, test_outputs = evaluate(model, test_loader, device)
            epoch_record["test_metrics"] = test_metrics
            print(
                f"epoch {epoch}/{args.epochs} | "
                f"test_auroc={test_metrics['auroc']:.4f} | "
                f"test_acc={test_metrics['acc']:.4f}"
            )
            candidate_score = test_metrics["auroc"] if args.save_best_metric == "auroc" else train_loss
            dump_json(test_outputs, output_dir / f"test_outputs_epoch_{epoch:03d}.json")

        history.append(epoch_record)
        dump_json(history, output_dir / "history.json")

        improved = candidate_score > best_score if args.save_best_metric == "auroc" else candidate_score < best_score
        if improved:
            best_score = candidate_score
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), best_model_path)

        if args.save_all_epochs:
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), output_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    summary = {
        "output_dir": str(output_dir),
        "best_model": str(best_model_path),
        "best_metric": args.save_best_metric,
        "best_score": best_score,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset) if test_dataset is not None else 0,
        "norm_stats": norm_stats,
    }
    dump_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
