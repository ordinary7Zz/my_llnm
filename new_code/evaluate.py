from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader

from models.modeling_LLNM_Net import CONFIGS, LLNM_Net
from new_code.dataset import LLNMDataset
from new_code.engine import evaluate
from new_code.utils import dump_json, load_pickle


def build_auroc_samples(outputs: dict[str, list]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    paths = outputs.get("paths", [])
    labels = outputs.get("labels", [])
    probs = outputs.get("probs", [])

    for path, label, prob in zip(paths, labels, probs):
        prob_class_1 = float(prob)
        prob_class_0 = float(1.0 - prob_class_1)
        samples.append(
            {
                "record_type": "sample",
                "image_file": str(path),
                "image_name": Path(str(path)).name,
                "true_label": int(label),
                "prob_class_0": prob_class_0,
                "prob_class_1": prob_class_1,
                "predicted_class": int(prob_class_1 >= 0.5),
                "confidence": float(max(prob_class_1, prob_class_0)),
            }
        )

    return samples


def derive_auroc_output_path(output_path: str) -> Path:
    path = Path(output_path)
    if path.suffix:
        return path.with_name(f"{path.stem}_auroc{path.suffix}")
    return path.with_name(f"{path.name}_auroc.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLNM-Net from JSON manifest")
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--norm_stats", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--patient_info", type=str, default=None)
    parser.add_argument("--radiomics_csv", type=str, default=None)
    parser.add_argument("--default_report", type=str, default="")
    parser.add_argument("--default_age", type=float, default=50.0)
    parser.add_argument("--default_sex", type=float, default=1.0)
    parser.add_argument("--shape_feature", type=str, default="original_shape2D_Elongation")
    parser.add_argument("--echo_feature", type=str, default="original_firstorder_Mean")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    norm_stats = load_pickle(args.norm_stats)
    dataset = LLNMDataset(
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
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    config = CONFIGS["LLNM_Net"]
    model = LLNM_Net(config, args.image_size, zero_head=True, num_classes=1)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    metrics, outputs = evaluate(model, dataloader, device)
    dump_json({"metrics": metrics, "outputs": outputs}, Path(args.output))
    dump_json(build_auroc_samples(outputs), derive_auroc_output_path(args.output))
    print(metrics)


if __name__ == "__main__":
    main()
