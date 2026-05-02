from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from new_code.utils import build_sample_records, dump_json, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned LLNM manifest from JSON labels")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--patient_info", type=str, default=None)
    parser.add_argument("--radiomics_csv", type=str, default=None)
    parser.add_argument("--default_report", type=str, default="")
    parser.add_argument("--default_age", type=float, default=50.0)
    parser.add_argument("--default_sex", type=float, default=1.0)
    parser.add_argument("--shape_feature", type=str, default="original_shape2D_Elongation")
    parser.add_argument("--echo_feature", type=str, default="original_firstorder_Mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = build_sample_records(
        manifest_path=args.input_json,
        image_root=args.image_root,
        default_report=args.default_report,
        default_age=args.default_age,
        default_sex=args.default_sex,
        patient_info_file=args.patient_info,
        radiomics_csv=args.radiomics_csv,
        shape_feature=args.shape_feature,
        echo_feature=args.echo_feature,
    )
    ensure_dir(Path(args.output_json).parent)
    dump_json(samples, args.output_json)
    print(f"saved {len(samples)} samples to {args.output_json}")


if __name__ == "__main__":
    main()
