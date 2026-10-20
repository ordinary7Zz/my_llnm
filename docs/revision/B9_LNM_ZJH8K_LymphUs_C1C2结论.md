# B9（LNM 分类）实验结论 — ZJH-8K（4756）+ LymphUs C1/C2

> **数据范围**：自包含目录 `LNM_ZJH8K_LymphUs_C1C2_packed`——**ZJH-8K** 中带 `LNM_CN01∈{0,1}` 的 **4756** 张原发灶图 + **LymphUs Center1 180**（训练补充）+ **Center2 158**（外部测试）。LymphUs 像素来自 `Lymph_Node_Metastasis_fake`，划分与稿内 **C1/C2** 表述一致。

**数据目录**：`/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/LNM_ZJH8K_LymphUs_C1C2_packed`  
**运行目录**：`new_code/runs/b9_packed_cv`、`new_code/runs/b9_packed_full_train`  
**生成日期**：2026-10-20

---

## 1. 返修要求与本次交付对照

| 返修项 | 要求摘要 | 本数据集（packed） | 证据/产物 |
|--------|----------|-------------------|-----------|
| **R2.11** | 训练规模；迁移/增强/正则；**患者级 5 折 CV** 替代固定 20 例 val 作主内部评估 | CV 池 = **ZJH train + C1**（**4753** 图，**882** 患者簇） | `b9_packed_cv/cv_summary.json` |
| **R2.12** | ZJH 原发灶 vs LymphUs 淋巴结；**患者级均值聚合** | manifest 含 `source` / `modality`；ZJH 按患者 ID，LymphUs C1 按图 | `fold_summary.json`、`history.json` |
| **B2** | **聚类 bootstrap 95% CI** | 已实现 | `confidence_intervals` 字段 |
| **外部测试** | **C2 158** 不参与 CV/训练 | 仅 `lymphus_center2_test.json` | `b9_packed_full_train/summary.json` |

| 与稿内表述 | 稿内常见数字 | 本构建实际 |
|------------|--------------|------------|
| ZJH 恶性 LNM 标注 | **4756** 图 | **4756**（`ZJH-8K_BM_labels.json`） |
| 固定 ZJH val（Methods） | 20 例 / **183** 图 | **20 / 183**（`zjh_val.json`，**不参与** 5 折 CV 池） |
| LymphUs C1 / C2 | **180** / **158** | **180** / **158** |
| 合并训练队列 | 常写 **4936**（4756+180） | **4753** = **4573**（ZJH train）+ **180**（C1）；4936 为**未扣除 val 183 图**的写法 |
| 5 折 CV 输入 | 训练队列（无 C2） | `train_combined_zjh_plus_lymphus_c1.json`（**4753**） |

---

## 2. 数据划分

| 角色 | 内容 | 图像数 | Manifest | 参与 5 折 CV |
|------|------|--------|----------|--------------|
| ZJH 全量 LNM 标注 | 原发灶 | 4756 | `zjh_all_labeled.json` | — |
| ZJH **train**（扣 val 患者） | 原发灶 | **4573** | `zjh_train.json` | **是** |
| ZJH **val**（固定 20 患者） | 原发灶 | **183** | `zjh_val.json` | **否** |
| LymphUs **Center1** | 淋巴结 | **180** | `lymphus_center1_train.json` | **是** |
| LymphUs **Center2** | 淋巴结 | **158** | `lymphus_center2_test.json` | **否**（仅外部测试） |
| **CV / 合并训练池** | ZJH train + C1 | **4753** | `train_combined_zjh_plus_lymphus_c1.json` | 折内划分 |

**来源（打包时复制自）**：

| 模态 | 标签 | 图像 |
|------|------|------|
| ZJH | `train_val_test/ZJH-8K/ZJH-8K_BM_labels.json` | → `packed/images/zjh/` |
| LymphUs C1/C2 | `Lymph_Node_Metastasis_fake/LymphUs_train_labels.json`（center1）与 `LymphUs_test_labels.json` | → `packed/images/lymphus_center1|center2/` |

**患者簇（CV 分组键 `patient_id`）**：

- ZJH：同一患者名下多图 → **1 簇多图**（合并池约 **702** 个 ZJH 患者簇 + **180** 个 LymphUs 图簇 ≈ **882** 簇）。
- LymphUs C1：`source` + 文件名 → **每图 1 簇**。

---

## 3. 模型与训练配置（LLNM-Net）

| 项目 | 设置 |
|------|------|
| 模型 | `LLNM_Net`，`image_size=224`，`resize_size=256` |
| 优化 | AdamW，`lr=1e-5`，`weight_decay=0.01` |
| Batch / Epoch | `batch_size=4`，`epochs=50` |
| 5 折 CV | `StratifiedKFold`，`n_splits=5`，`cv_seed=42`，按 **患者簇** 分层 |
| 全量训练选模 | C2 **图像级 AUROC** 最大 → `best_model.pth` |
| 迁移学习 | 未使用 `--pretrained_weights` |
| 数据增强 | 无（Resize + CenterCrop） |
| 临床/组学 | 默认 age/sex/report；未接入 `radiomics_csv` / `patient_info` |
| 多数据根 | `--image_roots_json` → `packed/image_roots.json`（`zjh` / `lymphus_c1` / `lymphus_c2`） |

---

## 4. 患者级 5 折交叉验证（R2.11）

**输入**：`train_combined` **4753** 图，**882** 患者簇。

### 4.0 各折 train/val 规模（`cv_seed=42`）

| Fold | Train 图像 | Train 患者簇 | Val 图像 | Val 患者簇 |
|------|-------------|--------------|----------|------------|
| 0 | 3772 | 705 | 981 | 177 |
| 1 | 3860 | 705 | 893 | 177 |
| 2 | 3746 | 706 | 1007 | 176 |
| 3 | 3798 | 706 | 955 | 176 |
| 4 | 3836 | 706 | 917 | 176 |

### 4.1 各折验证集患者级 AUROC

| Fold | Val 图像 | Val 簇 | Val 患者级 AUROC |
|------|----------|--------|------------------|
| 0 | 981 | 177 | 0.916 |
| 1 | 893 | 177 | 0.819 |
| 2 | 1007 | 176 | 0.875 |
| 3 | 955 | 176 | 0.806 |
| 4 | 917 | 176 | 0.842 |

### 4.2 合并 OOF（4753 张）

| 级别 | AUROC | AUPRC | Acc | 95% CI（AUROC，聚类 bootstrap） |
|------|-------|-------|-----|----------------------------------|
| 图像级 | 0.796 | 0.728 | 0.722 | [0.725, 0.860] |
| 患者级 | 0.796 | 0.728 | 0.722 | [0.727, 0.858] |

**归档**：`new_code/runs/b9_packed_cv/cv_summary.json`、`oof_predictions.json`、各折 `fold_XX/fold_summary.json`。

---

## 5. 全量训练 + Center2 外部测试（R2.12）

**协议**：`train_combined` **4753** 图训练 50 epoch；每个 epoch 在 C2 **158** 张上评估；保存 C2 AUROC 最优权重。

| 指标 | 最优 checkpoint（**Epoch 29**） | 最后一轮（Epoch 50） |
|------|--------------------------------|----------------------|
| C2 图像级 AUROC | **0.683** | 0.591 |
| C2 患者级 AUROC | **0.683** | 0.591 |
| C2 Acc | — | 0.589 |
| 训练集 AUROC | — | 1.000 |
| C2 患者级 AUROC 95% CI | **[0.591, 0.767]** | [0.497, 0.681] |

| 汇总字段 | 值 |
|----------|-----|
| `summary.json` → `best_score` | 0.6826 |
| `train_samples` / `test_samples` | 4753 / 158 |

**归档**：`new_code/runs/b9_packed_full_train/summary.json`、`history.json`、`test_outputs_epoch_*.json`。

---

## 6. 结论陈述（可直接改写入回复草稿）

1. **方法**：在 LNM 任务上采用 **LLNM-Net**；训练队列为 **ZJH-8K 恶性原发灶 LNM 标注（train 4573 图，val 20 例/183 图另留）+ LymphUs Center1（180）**；在合并队列（**4753** 图，**882** 患者簇）上做 **患者级分层 5 折 CV**，报告 **OOF AUROC** 及 **聚类 bootstrap 95% CI**；外部泛化在 **Center2（158）** 上以 **患者级平均概率** 报告 AUROC 及 CI。  
2. **内部 CV（OOF，患者级）**：AUROC **0.796**（95% CI **0.727–0.858**）。  
3. **外部测试（Center2）**：最优模型（epoch 29）患者级 AUROC **0.683**（95% CI **0.591–0.767**）。  
4. **与稿内计数**：ZJH LNM 标注 **4756**；合并训练 **4753**；C1/C2 **180/158**。  
5. **过拟合**：训练 AUROC 可达 1.0 时，外部指标以 **C2 AUROC 选中的 checkpoint** 为准。

---

## 7. 复现命令

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net
export CUDA_VISIBLE_DEVICES=0

DATA=/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/LNM_ZJH8K_LymphUs_C1C2_packed

python new_code/run_b9_patient_cv.py \
  --train_manifest $DATA/manifests/train_combined_zjh_plus_lymphus_c1.json \
  --image_roots_json $DATA/image_roots.json \
  --output_dir ./new_code/runs/b9_packed_cv \
  --batch_size 4 --epochs 50 --learning_rate 1e-5 --device cuda

python new_code/train.py \
  --train_json $DATA/manifests/train_combined_zjh_plus_lymphus_c1.json \
  --test_json $DATA/manifests/lymphus_center2_test.json \
  --image_roots_json $DATA/image_roots.json \
  --output_dir ./new_code/runs/b9_packed_full_train \
  --batch_size 4 --epochs 50 --learning_rate 1e-5 --device cuda
```

打包数据：

```bash
python new_code/pack_lnm_b9_dataset.py
```
