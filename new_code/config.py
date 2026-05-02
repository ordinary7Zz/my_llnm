from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    image_root: Path
    manifest_path: Path
    patient_info_file: Path | None = None
    radiomics_csv: Path | None = None
    default_report: str = ""
    default_age: float = 50.0
    default_sex: float = 1.0
    shape_feature: str = "original_shape2D_Elongation"
    echo_feature: str = "original_firstorder_Mean"
    max_report_tokens: int = 300


@dataclass(frozen=True)
class TrainConfig:
    train_manifest: Path
    test_manifest: Path | None
    image_root: Path
    output_dir: Path
    patient_info_file: Path | None = None
    radiomics_csv: Path | None = None
    pretrained_weights: Path | None = None
    default_report: str = ""
    default_age: float = 50.0
    default_sex: float = 1.0
    shape_feature: str = "original_shape2D_Elongation"
    echo_feature: str = "original_firstorder_Mean"
    image_size: int = 224
    resize_size: int = 256
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    num_workers: int = 0
    num_classes: int = 2
    max_report_tokens: int = 300
    device: str = "cuda"
    save_best_metric: str = "auroc"
