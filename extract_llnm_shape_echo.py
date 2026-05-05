from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"}
_EPS = 1e-6


@dataclass(frozen=True)
class Args:
    manifest_json: Path | None
    image_root: Path | None
    mask_dir: Path | None
    meta_image_dir: Path
    meta_mask_dir: Path
    nonmeta_image_dir: Path
    nonmeta_mask_dir: Path
    output_csv: Path
    path_mode: str
    relative_to: Path | None
    mask_threshold: int
    mask_suffix: str
    dilation_value: int
    load_stats_json: Path | None
    save_stats_json: Path | None
    skip_fail: bool
    limit: int | None
    label_key: str


@dataclass(frozen=True)
class EchoStats:
    mu_benign: float
    sigma_benign: float
    mu_malignant: float
    sigma_malignant: float
    benign_range: float
    malignant_range: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mu_benign": self.mu_benign,
            "sigma_benign": self.sigma_benign,
            "mu_malignant": self.mu_malignant,
            "sigma_malignant": self.sigma_malignant,
            "benign_range": self.benign_range,
            "malignant_range": self.malignant_range,
        }


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Extract LLNM-style shape and echo features from 2D images and masks."
    )
    parser.add_argument("--manifest_json", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--mask_dir", type=Path, default=None)
    parser.add_argument("--meta_image_dir", type=Path, required=True)
    parser.add_argument("--meta_mask_dir", type=Path, required=True)
    parser.add_argument("--nonmeta_image_dir", type=Path, required=True)
    parser.add_argument("--nonmeta_mask_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument(
        "--path_mode",
        choices=["relative", "basename", "absolute"],
        default="relative",
        help="How to write the filename column.",
    )
    parser.add_argument(
        "--relative_to",
        type=Path,
        default=None,
        help="Root used when path_mode=relative. Defaults to common parent of image dirs.",
    )
    parser.add_argument("--mask_threshold", type=int, default=0)
    parser.add_argument("--mask_suffix", type=str, default="")
    parser.add_argument("--dilation_value", type=int, default=3)
    parser.add_argument("--load_stats_json", type=Path, default=None)
    parser.add_argument("--save_stats_json", type=Path, default=None)
    parser.add_argument("--skip_fail", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label_key", type=str, default="LNM_CN01")
    raw = parser.parse_args()
    return Args(
        manifest_json=raw.manifest_json,
        image_root=raw.image_root,
        mask_dir=raw.mask_dir,
        meta_image_dir=raw.meta_image_dir,
        meta_mask_dir=raw.meta_mask_dir,
        nonmeta_image_dir=raw.nonmeta_image_dir,
        nonmeta_mask_dir=raw.nonmeta_mask_dir,
        output_csv=raw.output_csv,
        path_mode=raw.path_mode,
        relative_to=raw.relative_to,
        mask_threshold=raw.mask_threshold,
        mask_suffix=raw.mask_suffix,
        dilation_value=raw.dilation_value,
        load_stats_json=raw.load_stats_json,
        save_stats_json=raw.save_stats_json,
        skip_fail=bool(raw.skip_fail),
        limit=raw.limit,
        label_key=raw.label_key,
    )


def _list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    files = [path for path in sorted(image_dir.iterdir()) if path.suffix in _IMAGE_EXTS]
    return files


def _read_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape={arr.shape} ({path})")
    return arr


def _read_mask(path: Path, threshold: int) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale mask, got shape={arr.shape} ({path})")
    return (arr > threshold).astype(np.uint8)


def _find_mask(mask_dir: Path, image_path: Path, mask_suffix: str) -> Path:
    base = image_path.stem + mask_suffix
    for ext in _IMAGE_EXTS:
        candidate = mask_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Mask not found for {image_path.name} in {mask_dir}")


def _resolve_mask_path(mask_dir: Path, image_path: Path, mask_suffix: str) -> Path:
    relative = Path(image_path.name)
    direct = mask_dir / relative
    if direct.exists():
        return direct
    return _find_mask(mask_dir, image_path, mask_suffix)


def _normalize_to_255(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max - arr_min <= _EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr_min) / (arr_max - arr_min) * 255.0


def _resize_array(arr: np.ndarray, output_shape: tuple[int, int], resample: int) -> np.ndarray:
    image = Image.fromarray(arr)
    resized = image.resize((output_shape[1], output_shape[0]), resample=resample)
    return np.asarray(resized)


def _resize_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    resized = _resize_array((mask > 0).astype(np.uint8) * 255, output_shape, resample=Image.NEAREST)
    return (resized > 0).astype(np.uint8)


def _dilate_binary(mask: np.ndarray, dilation_value: int) -> np.ndarray:
    radius = max(int(dilation_value), 1)
    padded = np.pad(mask.astype(np.uint8), radius, mode="constant")
    output = np.zeros_like(mask, dtype=np.uint8)
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            window = padded[row : row + 2 * radius + 1, col : col + 2 * radius + 1]
            output[row, col] = 1 if np.any(window > 0) else 0
    return output


def _compute_llnm_ratio(mask: np.ndarray, dilation_value: int) -> float:
    resized = _resize_array((mask > 0).astype(np.uint8) * 255, (256, 256), resample=Image.NEAREST).astype(np.float32)
    dilated = _dilate_binary((resized > 0).astype(np.uint8), dilation_value).astype(np.float32) * 255.0
    dilated = _normalize_to_255(dilated)
    rows, cols = np.where(dilated > 125.0)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("No foreground found after dilation for shape ratio")
    height = int(rows.max() - rows.min() + 1)
    width = int(cols.max() - cols.min() + 1)
    return float(width / max(height, 1))


def _compute_echo_features(image: np.ndarray, mask: np.ndarray, dilation_value: int) -> tuple[float, float, float]:
    if image.shape != mask.shape:
        mask = _resize_mask(mask, image.shape)

    if int(mask.sum()) == 0:
        raise ValueError("Empty mask (no foreground pixels)")

    dilated = _dilate_binary(mask.astype(np.uint8), dilation_value)
    ring = np.logical_and(dilated > 0, mask == 0)
    if int(ring.sum()) == 0:
        raise ValueError("Empty boundary ring after dilation")

    nodule_values = image[mask > 0]
    bound_values = image[ring]
    if nodule_values.size == 0:
        raise ValueError("No nodule pixels available for echo feature")
    if bound_values.size == 0:
        raise ValueError("No boundary pixels available for echo feature")

    nodule_echo = float(np.mean(nodule_values))
    bound_echo = float(np.mean(bound_values))
    echo_sub = float(bound_echo - nodule_echo)
    return nodule_echo, bound_echo, echo_sub


def _pdf(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    sigma = max(float(sigma), _EPS)
    coefficient = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -((x - mu) ** 2) / (2.0 * sigma**2)
    return coefficient * np.exp(exponent)


def _max_pdf_difference(mu_a: float, sigma_a: float, mu_b: float, sigma_b: float, favor: str) -> float:
    sigma = max(sigma_a, sigma_b, _EPS)
    left = min(mu_a, mu_b) - 3.0 * sigma
    right = max(mu_a, mu_b) + 3.0 * sigma
    grid = np.linspace(left, right, 20001, dtype=np.float64)
    pdf_a = _pdf(grid, mu_a, sigma_a)
    pdf_b = _pdf(grid, mu_b, sigma_b)
    if favor == "a":
        diff = pdf_a - pdf_b
    else:
        diff = pdf_b - pdf_a
    return float(max(np.max(diff), _EPS))


def _fit_echo_stats(rows: list[dict[str, Any]]) -> EchoStats:
    benign = np.asarray([row["echo_sub"] for row in rows if row["label"] == 0], dtype=np.float64)
    malignant = np.asarray([row["echo_sub"] for row in rows if row["label"] == 1], dtype=np.float64)
    if benign.size == 0 or malignant.size == 0:
        raise ValueError("Both NonMeta(label=0) and Meta(label=1) samples are required to fit echo stats")

    mu_benign = float(np.mean(benign))
    sigma_benign = float(max(np.std(benign), _EPS))
    mu_malignant = float(np.mean(malignant))
    sigma_malignant = float(max(np.std(malignant), _EPS))
    benign_range = _max_pdf_difference(mu_benign, sigma_benign, mu_malignant, sigma_malignant, favor="a")
    malignant_range = _max_pdf_difference(mu_benign, sigma_benign, mu_malignant, sigma_malignant, favor="b")
    return EchoStats(
        mu_benign=mu_benign,
        sigma_benign=sigma_benign,
        mu_malignant=mu_malignant,
        sigma_malignant=sigma_malignant,
        benign_range=benign_range,
        malignant_range=malignant_range,
    )


def _load_echo_stats(path: Path) -> EchoStats:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EchoStats(
        mu_benign=float(payload["mu_benign"]),
        sigma_benign=float(payload["sigma_benign"]),
        mu_malignant=float(payload["mu_malignant"]),
        sigma_malignant=float(payload["sigma_malignant"]),
        benign_range=float(payload["benign_range"]),
        malignant_range=float(payload["malignant_range"]),
    )


def _compute_p_norm_echo(echo_sub: float, stats: EchoStats) -> float:
    p_malignant = float(_pdf(echo_sub, stats.mu_malignant, stats.sigma_malignant))
    p_benign = float(_pdf(echo_sub, stats.mu_benign, stats.sigma_benign))
    p_echo = p_malignant - p_benign
    if p_echo > 0:
        value = 0.5 + p_echo / (2.0 * max(stats.malignant_range, _EPS))
    else:
        value = 0.5 + p_echo / (2.0 * max(stats.benign_range, _EPS))
    return float(np.clip(value, 0.0, 1.0))


def _common_relative_root(args: Args) -> Path:
    if args.relative_to is not None:
        return args.relative_to.resolve()
    common = os.path.commonpath([
        str(args.meta_image_dir.resolve()),
        str(args.nonmeta_image_dir.resolve()),
    ])
    return Path(common)


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload and isinstance(payload["items"], list):
        payload = payload["items"]
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a list or dict with items: {manifest_path}")
    return payload


def _format_filename(path: Path, path_mode: str, relative_root: Path | None) -> str:
    resolved = path.resolve()
    if path_mode == "absolute":
        return resolved.as_posix()
    if path_mode == "basename":
        return path.name
    if relative_root is None:
        return path.name
    try:
        return resolved.relative_to(relative_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _extract_split(
    image_dir: Path,
    mask_dir: Path,
    label: int,
    args: Args,
    relative_root: Path | None,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0
    images = _list_images(image_dir)
    if args.limit is not None:
        images = images[: args.limit]

    split_name = "Meta" if label == 1 else "NonMeta"
    print(f"Processing {len(images)} {split_name} images...")

    for index, image_path in enumerate(images):
        try:
            mask_path = _find_mask(mask_dir, image_path, args.mask_suffix)
            image = _read_gray(image_path)
            mask = _read_mask(mask_path, args.mask_threshold)
            ratio = _compute_llnm_ratio(mask, args.dilation_value)
            nodule_echo, bound_echo, echo_sub = _compute_echo_features(image, mask, args.dilation_value)
            rows.append(
                {
                    "filename": _format_filename(image_path, args.path_mode, relative_root),
                    "label": int(label),
                    "llnm_ratio": ratio,
                    "nodule_echo": nodule_echo,
                    "bound_echo": bound_echo,
                    "echo_sub": echo_sub,
                }
            )
        except Exception as exc:
            failures += 1
            message = f"[{split_name}-{index}] failed: {image_path.name} err={type(exc).__name__}: {exc}"
            if args.skip_fail:
                print(message)
                continue
            raise RuntimeError(message) from exc

    return rows, failures


def _extract_from_manifest(
    args: Args,
    relative_root: Path | None,
) -> tuple[list[dict[str, Any]], int]:
    if args.manifest_json is None or args.image_root is None or args.mask_dir is None:
        raise ValueError("manifest_json, image_root, and mask_dir are required in manifest mode")

    rows: list[dict[str, Any]] = []
    failures = 0
    records = _load_manifest(args.manifest_json)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Processing {len(records)} records from manifest...")

    for index, record in enumerate(records):
        try:
            if args.label_key not in record:
                raise KeyError(f"label key '{args.label_key}' not found")
            label = int(record[args.label_key])
            if label not in (0, 1):
                raise ValueError(f"unsupported label: {label}")

            filename_value = record.get("filename", record.get("image"))
            if filename_value is None:
                raise KeyError("record must contain 'filename' or 'image'")

            filename = str(filename_value).replace("\\", "/")
            image_path = args.image_root / filename
            if not image_path.exists():
                raise FileNotFoundError(f"image not found: {image_path}")

            mask_path = _resolve_mask_path(args.mask_dir, image_path, args.mask_suffix)
            image = _read_gray(image_path)
            mask = _read_mask(mask_path, args.mask_threshold)
            ratio = _compute_llnm_ratio(mask, args.dilation_value)
            nodule_echo, bound_echo, echo_sub = _compute_echo_features(image, mask, args.dilation_value)
            rows.append(
                {
                    "filename": _format_filename(image_path, args.path_mode, relative_root),
                    "label": int(label),
                    "llnm_ratio": ratio,
                    "nodule_echo": nodule_echo,
                    "bound_echo": bound_echo,
                    "echo_sub": echo_sub,
                }
            )
        except Exception as exc:
            failures += 1
            message = f"[manifest-{index}] failed: err={type(exc).__name__}: {exc}"
            if args.skip_fail:
                print(message)
                continue
            raise RuntimeError(message) from exc

    return rows, failures


def main() -> None:
    args = parse_args()
    manifest_mode = args.manifest_json is not None
    if manifest_mode:
        if args.image_root is None:
            raise ValueError("--image_root is required when using --manifest_json")
        if args.mask_dir is None:
            raise ValueError("--mask_dir is required when using --manifest_json")
        relative_root = args.relative_to.resolve() if args.path_mode == "relative" and args.relative_to else args.image_root.resolve()
        rows, failures = _extract_from_manifest(args, relative_root)
        meta_failures = failures
        nonmeta_failures = 0
    else:
        relative_root = _common_relative_root(args) if args.path_mode == "relative" else None
        meta_rows, meta_failures = _extract_split(
            image_dir=args.meta_image_dir,
            mask_dir=args.meta_mask_dir,
            label=1,
            args=args,
            relative_root=relative_root,
        )
        nonmeta_rows, nonmeta_failures = _extract_split(
            image_dir=args.nonmeta_image_dir,
            mask_dir=args.nonmeta_mask_dir,
            label=0,
            args=args,
            relative_root=relative_root,
        )
        rows = meta_rows + nonmeta_rows
    if not rows:
        raise ValueError("No features extracted")

    if args.load_stats_json is not None:
        stats = _load_echo_stats(args.load_stats_json)
        print(f"Loaded echo stats from {args.load_stats_json}")
    else:
        stats = _fit_echo_stats(rows)
        print(
            "Fitted echo stats: "
            f"mu_benign={stats.mu_benign:.6f}, sigma_benign={stats.sigma_benign:.6f}, "
            f"mu_malignant={stats.mu_malignant:.6f}, sigma_malignant={stats.sigma_malignant:.6f}"
        )

    for row in rows:
        row["p_norm_echo"] = _compute_p_norm_echo(row["echo_sub"], stats)

    fieldnames = [
        "filename",
        "label",
        "llnm_ratio",
        "nodule_echo",
        "bound_echo",
        "echo_sub",
        "p_norm_echo",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    if args.save_stats_json is not None:
        args.save_stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_stats_json.write_text(
            json.dumps(stats.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved echo stats to {args.save_stats_json}")

    print(
        "Done. "
        f"meta_images={len(meta_rows) if not manifest_mode else 0} "
        f"nonmeta_images={len(nonmeta_rows) if not manifest_mode else 0} "
        f"failures={meta_failures + nonmeta_failures} saved={args.output_csv}"
    )


if __name__ == "__main__":
    main()
