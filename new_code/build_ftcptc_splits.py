from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LABEL_MAP = {"PTC": 0, "FTC": 1}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(data: object, output_path: str) -> None:
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FTC/PTC train/test JSON labels from cropped images")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/FangDai_Thyroid_Ultrasound_Images_cropped",
        help="Root directory that contains FTC/ and PTC/ subdirectories",
    )
    parser.add_argument("--train_output", type=str, default="datasets/FangDai_Thyroid_Ultrasound_Images_cropped/train_labels.json")
    parser.add_argument("--test_output", type=str, default="datasets/FangDai_Thyroid_Ultrasound_Images_cropped/test_labels.json")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_samples(data_root: Path) -> list[dict]:
    samples: list[dict] = []

    for class_name, label in LABEL_MAP.items():
        class_dir = data_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"class directory not found: {class_dir}")

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            samples.append(
                {
                    "filename": image_path.relative_to(data_root).as_posix(),
                    "FTCPTC": label,
                }
            )

    if not samples:
        raise ValueError(f"no image files found under {data_root}")

    return samples


def stratified_split(samples: list[dict], test_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    grouped: dict[int, list[dict]] = {0: [], 1: []}
    for sample in samples:
        grouped[sample["FTCPTC"]].append(sample)

    rng = random.Random(seed)
    train_samples: list[dict] = []
    test_samples: list[dict] = []

    for label, label_samples in grouped.items():
        if not label_samples:
            raise ValueError(f"label {label} has no samples")

        shuffled = list(label_samples)
        rng.shuffle(shuffled)

        test_count = max(1, int(round(len(shuffled) * test_size)))
        if test_count >= len(shuffled):
            test_count = len(shuffled) - 1
        if test_count <= 0:
            raise ValueError(f"label {label} does not have enough samples for splitting")

        test_samples.extend(shuffled[:test_count])
        train_samples.extend(shuffled[test_count:])

    train_samples.sort(key=lambda item: item["filename"])
    test_samples.sort(key=lambda item: item["filename"])
    return train_samples, test_samples


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    samples = collect_samples(data_root)
    train_samples, test_samples = stratified_split(samples, test_size=args.test_size, seed=args.seed)

    ensure_dir(Path(args.train_output).parent)
    ensure_dir(Path(args.test_output).parent)
    dump_json(train_samples, args.train_output)
    dump_json(test_samples, args.test_output)

    print(f"total: {len(samples)}")
    print(f"train: {len(train_samples)}")
    print(f"test: {len(test_samples)}")


if __name__ == "__main__":
    main()
