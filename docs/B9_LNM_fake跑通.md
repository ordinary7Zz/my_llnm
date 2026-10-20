# B9（LNM）在 Lymph_Node_Metastasis_fake 上跑通

不含 ZJH-8K，仅用于验证 **5 折 CV + 外部 C2 评估** 流程。正式返修请改用 `train_combined_zjh_plus_lymphus_c1.json`。

## 数据

| 用途 | Manifest |
|------|----------|
| 5 折 CV 训练池 | `LNM_ThyroidXAgent_ZJH_LymphUs_fake/manifests/lymphus_center1_train.json`（180 张，center1） |
| 外部测试 C2 | `.../lymphus_center2_test_LLNM_only.json`（158 张） |
| 图像根目录 | `Lymph_Node_Metastasis_fake/images` |

LymphUs 每条图有独立 `patient_id`（`utils.resolve_patient_id` 按 **文件名** 区分，避免同一 case 号下 Benign/Malignant 标签冲突）。

## 一键运行（推荐）

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net
CUDA_VISIBLE_DEVICES=0 bash new_code/run_b9_lnm_fake.sh
```

快速试跑（2 折、1 epoch）：

```bash
bash new_code/run_b9_lnm_fake.sh --smoke
```

可通过环境变量改输出前缀与超参，例如 `RUN_TAG=v1 EPOCHS=30 BATCH_SIZE=2 bash new_code/run_b9_lnm_fake.sh`。

## 步骤 1：5 折交叉验证（B9 核心输出）

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net

DATA=/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/LNM_ThyroidXAgent_ZJH_LymphUs_fake

python new_code/run_b9_patient_cv.py \
  --train_manifest $DATA/manifests/lymphus_center1_train.json \
  --image_roots_json $DATA/image_roots.json \
  --output_dir ./new_code/runs/b9_fake_cv \
  --n_splits 5 \
  --epochs 50 \
  --learning_rate 1e-5 \
  --device cuda
```

查看：

- `./new_code/runs/b9_fake_cv/cv_summary.json` → `oof_metrics_image` / `oof_metrics_patient`
- `./new_code/runs/b9_fake_cv/oof_predictions.json`

## 步骤 2：全量 C1 训练 + C2 外部测试

```bash
IMG=/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/images

python new_code/train.py \
  --train_json $DATA/manifests/lymphus_center1_train_LLNM_only.json \
  --test_json $DATA/manifests/lymphus_center2_test_LLNM_only.json \
  --train_image_root $IMG \
  --test_image_root $IMG \
  --output_dir ./new_code/runs/b9_fake_full_train \
  --epochs 50 \
  --learning_rate 1e-5 \
  --device cuda

python new_code/evaluate.py \
  --test_json $DATA/manifests/lymphus_center2_test_LLNM_only.json \
  --image_root $IMG \
  --weights ./new_code/runs/b9_fake_full_train/best_model.pth \
  --norm_stats ./new_code/runs/b9_fake_full_train/train_norm_stats.pkl \
  --output ./new_code/runs/b9_fake_full_train/c2_eval.json \
  --device cuda
```

`c2_eval.json` 中 `metrics_patient` 为 **外部集患者级（此处每图一簇）AUROC + 聚类 bootstrap CI**。

## 与正式 B9 的差异

| 项目 | fake 跑通 | 论文 / 正式 B9 |
|------|-----------|----------------|
| CV 训练池 | 仅 C1 180 张 | ZJH train + C1（约 8309 张） |
| 外部测试 | 同为 C2 158 | 同为 C2 158 |
| 填稿 | 仅作流程与脚本验证 | 用正式合并 manifest 重跑后填 Table S5 |
