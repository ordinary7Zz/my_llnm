"""
为 main2 提供的辅助二分类推理服务：

- 淋巴结转移二分类：与 ``infer_llnm/infer_images.py`` 相同的 LLNM-Net 推理流程（经子进程调用，避免与根目录 ``models`` 包冲突）
- 转移病理亚型二分类：与 ``infer_resnet/infer_resnet_directory.py`` 相同的 ResNet-18 目录推理流程

通过函数对一批图像路径一次性推理，返回按绝对路径字符串索引的结果字典。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from infer_resnet.infer_resnet_directory import (
    ImageFolderInferenceDataset,
    build_model,
    default_eval_transform,
    load_weights,
    run_inference as resnet_run_inference,
)


def _resolve_path(project_root: Path, p: str | None) -> str | None:
    if p is None:
        return None
    s = str(p).strip()
    if not s or s.lower() == "null":
        return None
    path = Path(s)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return str(path)


def _merge_shared_model_cfg(auxiliary_cfg: dict, model_key: str) -> dict[str, Any]:
    """
    将 ``auxiliary_binary_classification.shared`` 与模型子块合并，后者同名字段覆盖前者。
    未使用 ``shared`` 的旧配置（所有键写在 ``llnm`` / ``pathology_subtype_resnet`` 内）仍有效。
    """
    shared = dict(auxiliary_cfg.get("shared") or {})
    specific = dict(auxiliary_cfg.get(model_key) or {})
    return {**shared, **specific}


def _torch_device(device: str) -> torch.device:
    d = (device or "auto").strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _build_unified_binary_rec(
    abs_path: str,
    pred_class: int,
    prob_0: float,
    prob_1: float,
    relative_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建统一二分类输出键。"""
    path_obj = Path(abs_path).resolve()
    rel = relative_path if relative_path else path_obj.name
    rec: dict[str, Any] = {
        "relative_path": str(rel),
        "filename": path_obj.name,
        "pred_class": int(pred_class),
        "prob_class_0": float(prob_0),
        "prob_class_1": float(prob_1),
    }
    if extra:
        rec.update(extra)
    return rec


def run_llnm_binary_for_image_paths(
    image_paths: list[Path],
    llnm_cfg: dict,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """
    对给定图像路径列表运行 LLNM-Net 二分类（多模态侧使用配置中的默认值）。

    返回: ``abs_path_str -> { "filename", "relative_path", "pred_class", "prob_class_0", "prob_class_1" }``
    """
    out: dict[str, dict[str, Any]] = {}
    if not image_paths:
        return out

    model_path = _resolve_path(project_root, llnm_cfg.get("model_path"))
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"LLNM 权重不存在或未配置: {llnm_cfg.get('model_path')}")

    norm_params = _resolve_path(project_root, llnm_cfg.get("norm_params_file"))

    default_shape_echo = llnm_cfg.get("default_shape_echo", [0.0, 0.0])
    if isinstance(default_shape_echo, str):
        default_shape_echo = [float(x.strip()) for x in default_shape_echo.split(",")]
    shape_tuple = (float(default_shape_echo[0]), float(default_shape_echo[1]))

    worker = Path(__file__).resolve().parent / "infer_llnm" / "batch_infer_worker.py"
    if not worker.is_file():
        raise FileNotFoundError(f"未找到 LLNM 子进程脚本: {worker}")

    dev = str(llnm_cfg.get("device", "cuda")).strip().lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    payload: dict[str, Any] = {
        "model_path": model_path,
        "batch_size": int(llnm_cfg.get("batch_size", 4)),
        "num_classes": int(llnm_cfg.get("num_classes", 2)),
        "device": dev,
        "threshold": float(llnm_cfg.get("threshold", 0.5)),
        "default_report": str(llnm_cfg.get("default_report", "") or ""),
        "default_age": float(llnm_cfg.get("default_age", 50.0)),
        "default_sex": float(llnm_cfg.get("default_sex", 1.0)),
        "default_shape_echo": [shape_tuple[0], shape_tuple[1]],
        "norm_params_file": norm_params,
        "records": [{"path": str(p.resolve()), "out_key": p.name} for p in image_paths],
    }

    proc = subprocess.run(
        [sys.executable, str(worker)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=str(project_root),
        encoding="utf-8",
    )
    raw_out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or raw_out
        raise RuntimeError(err or "LLNM 子进程失败")
    if not raw_out:
        raise RuntimeError("LLNM 子进程无输出")
    line = raw_out.splitlines()[-1]
    data = json.loads(line)
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "LLNM 子进程未知错误"))

    for row in data.get("results", []):
        key = str(Path(row["path"]).resolve())
        out[key] = _build_unified_binary_rec(
            abs_path=key,
            relative_path=str(row.get("relative_path", "") or ""),
            pred_class=int(row["pred_class"]),
            prob_0=float(row["prob_class_0"]),
            prob_1=float(row["prob_class_1"]),
        )
    return out


def run_resnet_binary_for_image_paths(
    image_paths: list[Path],
    resnet_cfg: dict,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """
    对给定图像路径列表运行 ResNet 多类/二分类（与 infer_resnet_directory 相同的预处理）。

    返回: ``abs_path_str -> { "filename", "relative_path", "pred_class", "prob_class_0", "prob_class_1" }``
    若 ``num_classes > 2``，额外输出 ``prob_class_2`` ... ``prob_class_n``。
    """
    out: dict[str, dict[str, Any]] = {}
    if not image_paths:
        return out

    pth_path = _resolve_path(project_root, resnet_cfg.get("pth_path"))
    if not pth_path or not os.path.isfile(pth_path):
        raise FileNotFoundError(f"ResNet 权重不存在或未配置: {resnet_cfg.get('pth_path')}")

    num_classes = int(resnet_cfg.get("num_classes", 2))
    architecture = str(resnet_cfg.get("architecture", "resnet"))
    batch_size = int(resnet_cfg.get("batch_size", 32))
    num_workers = int(resnet_cfg.get("num_workers", 4))
    device = _torch_device(str(resnet_cfg.get("device", "auto")))

    str_paths = [str(p.resolve()) for p in image_paths]
    transform = default_eval_transform()
    dataset = ImageFolderInferenceDataset(str_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes, architecture)
    load_weights(model, pth_path, device)
    model = model.to(device)

    df: pd.DataFrame = resnet_run_inference(model, loader, device, num_classes)
    for _, row in df.iterrows():
        p = str(Path(row["path"]).resolve())
        prob_0 = float(row["prob_0"]) if "prob_0" in row else 0.0
        prob_1 = float(row["prob_1"]) if "prob_1" in row else (1.0 - prob_0)
        rec: dict[str, Any] = _build_unified_binary_rec(
            abs_path=p,
            relative_path=row.get("filename", os.path.basename(row["path"])),
            pred_class=int(row["pred_class"]),
            prob_0=prob_0,
            prob_1=prob_1,
        )
        for c in range(2, num_classes):
            rec[f"prob_class_{c}"] = float(row[f"prob_{c}"])
        out[p] = rec
    return out


def compute_auxiliary_binary_predictions(
    image_paths: list[Path],
    auxiliary_cfg: dict,
    project_root: Path,
) -> dict[str, dict[str, Any]] | None:
    """
    若配置中 ``enable`` 为假，返回 ``None``。

    若为真，依次运行 ResNet 与 LLNM（各自失败时打印警告并跳过该分支）。

    返回: ``abs_path_str -> { "llnm_lymph_node": {...} | None, "pathology_subtype_resnet": {...} | None }``
    """
    if not auxiliary_cfg.get("enable", False):
        return None

    merged: dict[str, dict[str, Any]] = {}
    for p in image_paths:
        merged[str(p.resolve())] = {
            "llnm_lymph_node": None,
            "pathology_subtype_resnet": None,
        }

    llnm_cfg = _merge_shared_model_cfg(auxiliary_cfg, "llnm")
    res_cfg = _merge_shared_model_cfg(auxiliary_cfg, "pathology_subtype_resnet")

    if res_cfg.get("pth_path"):
        try:
            rmap = run_resnet_binary_for_image_paths(image_paths, res_cfg, project_root)
            for k, v in rmap.items():
                if k in merged:
                    merged[k]["pathology_subtype_resnet"] = v
        except Exception as e:
            print(f"⚠️  转移病理亚型 ResNet 推理失败（已跳过）: {e}")

    if llnm_cfg.get("model_path"):
        try:
            lmap = run_llnm_binary_for_image_paths(image_paths, llnm_cfg, project_root)
            for k, v in lmap.items():
                if k in merged:
                    merged[k]["llnm_lymph_node"] = v
        except Exception as e:
            print(f"⚠️  淋巴结转移 LLNM 推理失败（已跳过）: {e}")

    return merged


def attach_auxiliary_to_result(
    result_dict: dict[str, Any],
    image_file: Path,
    auxiliary_by_path: dict[str, dict[str, Any]] | None,
) -> None:
    """将辅助二分类结果原地写入 ``result_dict``（字段名固定，便于下游解析）。"""
    if not auxiliary_by_path:
        return
    block = auxiliary_by_path.get(str(image_file.resolve()))
    if not block:
        return
    result_dict["auxiliary_binary_classification"] = {
        "llnm_lymph_node_metastasis": block.get("llnm_lymph_node"),
        "pathology_subtype_resnet": block.get("pathology_subtype_resnet"),
    }
