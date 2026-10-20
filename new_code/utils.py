from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


VALID_LABELS = {0, 1}


def load_manifest(manifest_path: str | Path) -> list[dict]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            data = data["items"]
        else:
            raise ValueError(f"JSON manifest format not supported: {path}")

    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list or dict with items: {path}")

    return data


def filter_labeled_samples(records: list[dict], label_key: str = "LNM_CN01") -> list[dict]:
    filtered = []
    for record in records:
        label = record.get(label_key)
        if label in VALID_LABELS:
            filtered.append(record)
    return filtered


def extract_name_from_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]

    match = re.match(r"^([^_]+)__", stem)
    if match:
        return match.group(1)

    match = re.match(r"^([^_]+)_", stem)
    if match:
        return match.group(1)

    return stem


def load_patient_info(patient_info_file: str | Path | None, default_age: float, default_sex: float) -> dict[str, dict[str, float]]:
    if patient_info_file is None:
        return {}

    path = Path(patient_info_file)
    if not path.exists():
        raise FileNotFoundError(f"patient info file not found: {path}")

    patient_df = pd.read_excel(path)
    patient_info_dict: dict[str, dict[str, float]] = {}

    for _, row in patient_df.iterrows():
        name = str(row["姓名"]).strip() if "姓名" in row and pd.notna(row["姓名"]) else None
        if not name:
            continue

        age = row["年龄"] if "年龄" in row and pd.notna(row["年龄"]) else default_age
        sex = row["性别"] if "性别" in row and pd.notna(row["性别"]) else default_sex

        if isinstance(sex, str):
            sex = 0 if sex in ["女", "F", "f", "female", "Female"] else 1

        patient_info_dict[name] = {"age": float(age), "sex": float(sex)}

    return patient_info_dict


def load_radiomics_features(
    radiomics_csv: str | Path | None,
    shape_feature: str,
    echo_feature: str,
) -> dict[str, dict[str, float]]:
    if radiomics_csv is None:
        return {}

    path = Path(radiomics_csv)
    if not path.exists():
        raise FileNotFoundError(f"radiomics csv not found: {path}")

    radiomics_df = pd.read_csv(path)
    radiomics_dict: dict[str, dict[str, float]] = {}

    for _, row in radiomics_df.iterrows():
        filename = str(row["filename"]).replace("\\", "/") if "filename" in row and pd.notna(row["filename"]) else None
        if not filename:
            continue

        shape_val = row[shape_feature] if shape_feature in row and pd.notna(row[shape_feature]) else 0.0
        echo_val = row[echo_feature] if echo_feature in row and pd.notna(row[echo_feature]) else 0.0
        mask_empty_val = row["mask_empty"] if "mask_empty" in row and pd.notna(row["mask_empty"]) else 0.0
        radiomics_dict[filename] = {"shape": float(shape_val), "echo": float(echo_val), "mask_empty": int(mask_empty_val)}
        radiomics_dict[os.path.basename(filename)] = {"shape": float(shape_val), "echo": float(echo_val), "mask_empty": int(mask_empty_val)}

    return radiomics_dict


def resolve_image_path(image_root: str | Path, filename: str) -> Path:
    return Path(image_root) / Path(filename)


def load_image_roots_json(path: str | Path) -> dict[str, Path]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"image_roots json must be a dict: {path}")
    return {str(k): Path(v) for k, v in data.items()}


def is_lymphus_style_filename(filename: str) -> bool:
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    return "_center1" in stem or "_center2" in stem


def resolve_patient_id(record: dict, filename: str, image_root_key: str) -> str:
    source = str(record.get("source") or "")
    if source.startswith("lymphus") or is_lymphus_style_filename(filename):
        # LymphUs fake/C1: same numeric case id can have both Benign/Malignant images with different labels.
        prefix = source if source.startswith("lymphus") else "lymphus"
        return f"{prefix}::{filename}"
    patient_id = record.get("patient_id")
    if patient_id is None:
        patient_id = extract_name_from_filename(filename)
    prefix = source or image_root_key
    return f"{prefix}::{patient_id}"


def build_sample_records(
    manifest_path: str | Path,
    image_root: str | Path,
    default_report: str = "",
    default_age: float = 50.0,
    default_sex: float = 1.0,
    patient_info_file: str | Path | None = None,
    radiomics_csv: str | Path | None = None,
    shape_feature: str = "original_shape2D_Elongation",
    echo_feature: str = "original_firstorder_Mean",
    label_key: str = "LNM_CN01",
    image_roots: dict[str, Path] | None = None,
    default_image_root_key: str = "zjh",
) -> list[dict]:
    records = filter_labeled_samples(load_manifest(manifest_path), label_key=label_key)
    return build_sample_records_from_list(
        records=records,
        image_root=image_root,
        default_report=default_report,
        default_age=default_age,
        default_sex=default_sex,
        patient_info_file=patient_info_file,
        radiomics_csv=radiomics_csv,
        shape_feature=shape_feature,
        echo_feature=echo_feature,
        label_key=label_key,
        image_roots=image_roots,
        default_image_root_key=default_image_root_key,
    )


def build_sample_records_from_list(
    records: list[dict],
    image_root: str | Path,
    default_report: str = "",
    default_age: float = 50.0,
    default_sex: float = 1.0,
    patient_info_file: str | Path | None = None,
    radiomics_csv: str | Path | None = None,
    shape_feature: str = "original_shape2D_Elongation",
    echo_feature: str = "original_firstorder_Mean",
    label_key: str = "LNM_CN01",
    image_roots: dict[str, Path] | None = None,
    default_image_root_key: str = "zjh",
) -> list[dict]:
    patient_info_dict = load_patient_info(patient_info_file, default_age, default_sex)
    radiomics_dict = load_radiomics_features(radiomics_csv, shape_feature, echo_feature)

    samples: list[dict] = []
    missing_images: list[str] = []

    for record in records:
        filename_value = record.get("filename", record.get("image"))
        if filename_value is None:
            raise KeyError("record must contain 'filename' or 'image'")
        filename = str(filename_value).replace("\\", "/")

        root_key = str(record.get("image_root_key", default_image_root_key))
        if image_roots is not None:
            if root_key not in image_roots:
                raise KeyError(f"unknown image_root_key '{root_key}' in record: {filename}")
            resolved_root = image_roots[root_key]
        else:
            resolved_root = Path(image_root)

        image_path = resolve_image_path(resolved_root, filename)
        if not image_path.exists():
            missing_images.append(filename)
            continue

        patient_name = extract_name_from_filename(filename)
        patient_info = patient_info_dict.get(patient_name, {})
        age = float(record.get("age", patient_info.get("age", default_age)))
        sex = float(record.get("sex", patient_info.get("sex", default_sex)))
        report = str(record.get("report", default_report) or default_report)

        radiomics = radiomics_dict.get(filename) or radiomics_dict.get(os.path.basename(filename)) or {}
        shape_val = float(record.get("shape", radiomics.get("shape", 0.0)))
        echo_val = float(record.get("echo", radiomics.get("echo", 0.0)))
        mask_empty_val = int(record.get("mask_empty", radiomics.get("mask_empty", 0)))

        samples.append(
            {
                "filename": filename,
                "image": filename,
                "image_root": str(resolved_root),
                "image_root_key": root_key,
                "patient_id": resolve_patient_id(record, filename, root_key),
                "LNM_CN01": int(record[label_key]),
                "report": report,
                "age": age,
                "sex": sex,
                "shape": shape_val,
                "echo": echo_val,
                "mask_empty": mask_empty_val,
                "source": record.get("source"),
            }
        )

    if missing_images:
        print(f"skipped {len(missing_images)} missing images")

    return samples


def compute_normalization_stats(samples: list[dict]) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot compute normalization stats on empty samples")

    ages = np.array([sample["age"] for sample in samples], dtype=np.float32)
    shapes = np.array([sample["shape"] for sample in samples], dtype=np.float32)
    echos = np.array([sample["echo"] for sample in samples], dtype=np.float32)

    return {
        "age_mean": float(ages.mean()),
        "age_std": float(max(ages.std(), 1e-6)),
        "shape_mean": float(shapes.mean()),
        "shape_std": float(max(shapes.std(), 1e-6)),
        "echo_mean": float(echos.mean()),
        "echo_std": float(max(echos.std(), 1e-6)),
    }


def apply_normalization(sample: dict, norm_stats: dict[str, float]) -> dict:
    normalized = dict(sample)
    normalized["age"] = (sample["age"] - norm_stats["age_mean"]) / norm_stats["age_std"]
    normalized["shape"] = (sample["shape"] - norm_stats["shape_mean"]) / norm_stats["shape_std"]
    normalized["echo"] = (sample["echo"] - norm_stats["echo_mean"]) / norm_stats["echo_std"]
    return normalized


def save_pickle(data: object, output_path: str | Path) -> None:
    path = Path(output_path)
    with path.open("wb") as f:
        pickle.dump(data, f)


def load_pickle(input_path: str | Path):
    path = Path(input_path)
    with path.open("rb") as f:
        return pickle.load(f)


def calculate_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = 0.5

    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auprc"] = 0.0

    return metrics


def calculate_binary_metrics_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.size == 0:
        return {}

    rng = np.random.default_rng(random_state)
    metric_names = tuple(calculate_binary_metrics(y_true, y_prob, threshold=threshold).keys())
    bootstrap_metrics = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, y_true.size, size=y_true.size)
        sampled_metrics = calculate_binary_metrics(y_true[indices], y_prob[indices], threshold=threshold)
        for name in metric_names:
            bootstrap_metrics[name].append(sampled_metrics[name])

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    return {
        name: {
            "lower": float(np.percentile(values, lower_q)),
            "upper": float(np.percentile(values, upper_q)),
        }
        for name, values in bootstrap_metrics.items()
    }


def aggregate_predictions_by_patient(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: list[str] | np.ndarray,
    strategy: str = "mean",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    patient_ids = list(patient_ids)

    grouped: dict[str, list[tuple[int, float]]] = {}
    for label, prob, pid in zip(y_true, y_prob, patient_ids):
        grouped.setdefault(pid, []).append((int(label), float(prob)))

    ordered_ids = sorted(grouped.keys())
    patient_labels: list[int] = []
    patient_probs: list[float] = []
    for pid in ordered_ids:
        pairs = grouped[pid]
        labels = [p[0] for p in pairs]
        if len(set(labels)) > 1:
            raise ValueError(f"inconsistent labels for patient {pid}: {set(labels)}")
        probs = [p[1] for p in pairs]
        if strategy == "mean":
            agg_prob = float(np.mean(probs))
        elif strategy == "max":
            agg_prob = float(np.max(probs))
        else:
            raise ValueError(f"unsupported aggregation strategy: {strategy}")
        patient_labels.append(labels[0])
        patient_probs.append(agg_prob)

    return np.asarray(patient_labels, dtype=int), np.asarray(patient_probs, dtype=float), ordered_ids


def calculate_binary_metrics_confidence_intervals_clustered(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: list[str] | np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    aggregation: str = "mean",
) -> dict[str, dict[str, float]]:
    patient_ids = np.asarray(patient_ids)
    unique_patients = np.unique(patient_ids)
    if unique_patients.size == 0:
        return {}

    rng = np.random.default_rng(random_state)
    patient_to_indices: dict[str, np.ndarray] = {}
    for pid in unique_patients:
        patient_to_indices[str(pid)] = np.flatnonzero(patient_ids == pid)

    y_patient, p_patient, _ = aggregate_predictions_by_patient(y_true, y_prob, patient_ids, strategy=aggregation)
    metric_names = tuple(calculate_binary_metrics(y_patient, p_patient, threshold=threshold).keys())
    bootstrap_metrics = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_patients, size=unique_patients.size, replace=True)
        boot_labels: list[int] = []
        boot_probs: list[float] = []
        for pid in sampled:
            indices = patient_to_indices[str(pid)]
            labels = y_true[indices]
            probs = y_prob[indices]
            if len(set(labels.tolist())) > 1:
                raise ValueError(f"inconsistent labels within patient cluster {pid}")
            if aggregation == "mean":
                boot_probs.append(float(np.mean(probs)))
            else:
                boot_probs.append(float(np.max(probs)))
            boot_labels.append(int(labels[0]))
        sampled_metrics = calculate_binary_metrics(
            np.asarray(boot_labels, dtype=int),
            np.asarray(boot_probs, dtype=float),
            threshold=threshold,
        )
        for name in metric_names:
            bootstrap_metrics[name].append(sampled_metrics[name])

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    return {
        name: {
            "lower": float(np.percentile(values, lower_q)),
            "upper": float(np.percentile(values, upper_q)),
        }
        for name, values in bootstrap_metrics.items()
    }


def summarize_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: list[str] | np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    patient_aggregation: str = "mean",
) -> dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    metrics_image = calculate_binary_metrics(y_true, y_prob)
    metrics_image["confidence_intervals"] = calculate_binary_metrics_confidence_intervals(
        y_true,
        y_prob,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    y_patient, p_patient, ordered_patient_ids = aggregate_predictions_by_patient(
        y_true,
        y_prob,
        patient_ids,
        strategy=patient_aggregation,
    )
    metrics_patient = calculate_binary_metrics(y_patient, p_patient)
    metrics_patient["confidence_intervals"] = calculate_binary_metrics_confidence_intervals_clustered(
        y_true,
        y_prob,
        patient_ids,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        aggregation=patient_aggregation,
    )
    metrics_patient["n_patients"] = int(len(ordered_patient_ids))

    return {
        "metrics_image": metrics_image,
        "metrics_patient": metrics_patient,
        "patient_predictions": [
            {
                "patient_id": pid,
                "true_label": int(label),
                "prob_class_1": float(prob),
            }
            for pid, label, prob in zip(ordered_patient_ids, y_patient, p_patient)
        ],
    }


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_json(data: object, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.long().view(-1)
    target = torch.zeros(labels.shape[0], num_classes, device=labels.device)
    target.scatter_(1, labels.unsqueeze(1), 1.0)
    return target
