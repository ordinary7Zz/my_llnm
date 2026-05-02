import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG")


def _list_images(image_dir: str) -> List[str]:
    files = []
    for name in os.listdir(image_dir):
        if name.endswith(_IMAGE_EXTS):
            files.append(os.path.join(image_dir, name))
    return sorted(files)


def _read_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={arr.shape} ({path})")
    return arr


def _read_mask(path: str, threshold: int) -> np.ndarray:
    m = Image.open(path).convert("L")
    arr = np.asarray(m)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={arr.shape} ({path})")
    return (arr > threshold).astype(np.uint8)


def _find_mask(mask_dir: str, image_path: str, mask_suffix: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    cand_base = base + (mask_suffix or "")

    for ext in _IMAGE_EXTS:
        cand = os.path.join(mask_dir, cand_base + ext)
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Mask not found for image '{image_path}'. Looked for '{cand_base}'+({', '.join(_IMAGE_EXTS)}) in {mask_dir}"
    )





def _to_sitk(image: np.ndarray, mask: np.ndarray, spacing: Tuple[float, float]):
    import SimpleITK as sitk

    img_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    msk_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))

    img_sitk.SetSpacing(spacing)
    msk_sitk.SetSpacing(spacing)

    return img_sitk, msk_sitk


def _extract_one(
    extractor,
    image_path: str,
    mask_path: str,
    label_value: int,
    mask_threshold: int,
    spacing: Tuple[float, float],
) -> Dict[str, object]:
    image = _read_gray(image_path)
    mask = _read_mask(mask_path, threshold=mask_threshold)

    if image.shape != mask.shape:
        raise ValueError(f"Image/mask size mismatch: image={image.shape}, mask={mask.shape}")

    if int(mask.sum()) == 0:
        raise ValueError("Empty mask (no foreground pixels)")

    # PyRadiomics expects SimpleITK images
    img_sitk, msk_sitk = _to_sitk(image, mask, spacing=spacing)

    result = extractor.execute(img_sitk, msk_sitk, label=1)

    # drop diagnostics; keep numeric features
    feats = {k: v for k, v in result.items() if not str(k).startswith("diagnostics_")}
    feats["filename"] = os.path.basename(image_path)
    feats["label"] = int(label_value)
    return feats


@dataclass
class Args:
    meta_image_dir: str
    meta_mask_dir: str
    nonmeta_image_dir: str
    nonmeta_mask_dir: str
    output_csv: str
    params: str
    mask_threshold: int
    mask_suffix: str
    spacing_x: float
    spacing_y: float
    skip_fail: bool
    limit: Optional[int]


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="2D pyradiomics feature extraction for Meta vs NonMeta classification using segmentation masks as ROI."
    )
    p.add_argument("--meta_image_dir", type=str, required=True, help="directory of Meta 2D images (png/jpg)")
    p.add_argument("--meta_mask_dir", type=str, required=True, help="directory of Meta 2D masks (png/jpg)")
    p.add_argument("--nonmeta_image_dir", type=str, required=True, help="directory of NonMeta 2D images (png/jpg)")
    p.add_argument("--nonmeta_mask_dir", type=str, required=True, help="directory of NonMeta 2D masks (png/jpg)")

    p.add_argument("--output_csv", type=str, required=True, help="output features CSV")

    p.add_argument(
        "--params",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "radiomics_2d.yaml"),
        help="pyradiomics parameter YAML",
    )

    p.add_argument("--mask_threshold", type=int, default=0, help="mask > threshold is treated as foreground")
    p.add_argument("--mask_suffix", type=str, default="", help="optional suffix added to image basename to find mask")

    # no real spacing; keep consistent pseudo spacing
    p.add_argument("--spacing_x", type=float, default=1.0)
    p.add_argument("--spacing_y", type=float, default=1.0)

    p.add_argument("--skip_fail", action="store_true", help="skip failed cases instead of stopping")
    p.add_argument("--limit", type=int, default=None, help="debug: only process first N images per class")

    a = p.parse_args()
    return Args(
        meta_image_dir=a.meta_image_dir,
        meta_mask_dir=a.meta_mask_dir,
        nonmeta_image_dir=a.nonmeta_image_dir,
        nonmeta_mask_dir=a.nonmeta_mask_dir,
        output_csv=a.output_csv,
        params=a.params,
        mask_threshold=a.mask_threshold,
        mask_suffix=a.mask_suffix,
        spacing_x=a.spacing_x,
        spacing_y=a.spacing_y,
        skip_fail=bool(a.skip_fail),
        limit=a.limit,
    )


def main() -> None:
    args = parse_args()

    from radiomics import featureextractor

    # Load images for both classes
    meta_images = _list_images(args.meta_image_dir)
    nonmeta_images = _list_images(args.nonmeta_image_dir)
    
    if args.limit is not None:
        meta_images = meta_images[: args.limit]
        nonmeta_images = nonmeta_images[: args.limit]

    extractor = featureextractor.RadiomicsFeatureExtractor(args.params)

    rows: List[Dict[str, object]] = []
    failures = 0

    # Process Meta images (label=1)
    print(f"Processing {len(meta_images)} Meta images...")
    for idx, img_path in enumerate(meta_images):
        fname = os.path.basename(img_path)
        try:
            mask_path = _find_mask(args.meta_mask_dir, img_path, mask_suffix=args.mask_suffix)
            feats = _extract_one(
                extractor,
                image_path=img_path,
                mask_path=mask_path,
                label_value=1,  # Meta = 1
                mask_threshold=args.mask_threshold,
                spacing=(args.spacing_x, args.spacing_y),
            )
            rows.append(feats)
        except Exception as e:
            failures += 1
            msg = f"[Meta-{idx}] failed: {fname} err={type(e).__name__}: {e}"
            if args.skip_fail:
                print(msg)
                continue
            raise RuntimeError(msg) from e

    # Process NonMeta images (label=0)
    print(f"Processing {len(nonmeta_images)} NonMeta images...")
    for idx, img_path in enumerate(nonmeta_images):
        fname = os.path.basename(img_path)
        try:
            mask_path = _find_mask(args.nonmeta_mask_dir, img_path, mask_suffix=args.mask_suffix)
            feats = _extract_one(
                extractor,
                image_path=img_path,
                mask_path=mask_path,
                label_value=0,  # NonMeta = 0
                mask_threshold=args.mask_threshold,
                spacing=(args.spacing_x, args.spacing_y),
            )
            rows.append(feats)
        except Exception as e:
            failures += 1
            msg = f"[NonMeta-{idx}] failed: {fname} err={type(e).__name__}: {e}"
            if args.skip_fail:
                print(msg)
                continue
            raise RuntimeError(msg) from e

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(
        "Done. "
        f"meta_images={len(meta_images)} nonmeta_images={len(nonmeta_images)} "
        f"extracted={len(df)} failures={failures} saved={args.output_csv}"
    )


if __name__ == "__main__":
    main()
