# B9（LNM 分类）实验结论 — 以 Lymph_Node_Metastasis_fake 为训练数据

> **范围说明（本稿前提）**：将 **LymphUs fake** 作为 LNM 任务的训练与评估数据载体——**Center1（180 张）** 为训练队列，**Center2（158 张）** 为外部测试；**不纳入 ZJH 原发灶**。在此前提下汇总已完成的 B9 流程与数值，供返修 Methods / 回复 R2.11–R2.12 起草；若后续改为稿内 **ZJH train + C1** 合并队列，需用同一脚本重跑并替换下表数字。

**运行目录**：`new_code/runs/b9_fake_cv`、`new_code/runs/b9_fake_full_train`  
**数据目录**：`Classifaction_Data/LNM_ThyroidXAgent_ZJH_LymphUs_fake`  
**生成日期**：2026-10-20

---

## 1. 返修要求与本次交付对照

| 返修项 | 要求摘要 | 本次（fake 前提）状态 | 证据/产物 |
|--------|----------|----------------------|-----------|
| **R2.11** | 训练规模写清；说明迁移/增强/正则；用 **患者级 5 折 CV** 替代固定小 val 做过拟合评估 | **已做**（CV 队列 = C1 180 张，5 折分层） | `b9_fake_cv/cv_summary.json`、`cv_config.json` |
| **R2.12** | LymphUs 淋巴结图 LNM 定义；**患者级概率聚合**（均值） | **已做**（fake 上每图一患者 ID；聚合=均值） | CV / `train.py` 日志与 `history.json` 中 `test_metrics_patient` |
| **B2 交叉** | 指标 **95% CI** 为 **患者（聚类）bootstrap** | **已做** | OOF 与外部测试均含 `confidence_intervals` |
| **外部验证** | LymphUs **Center2** 不参与训练 | **已做**（C1 训 → C2 测） | `b9_fake_full_train/best_model.pth`、`summary.json` |

| 与 ThyroidXAgent 稿内数字的差异 | 稿内目标 | 本实验实际 |
|--------------------------------|----------|------------|
| 训练图像数 | 4936（4756 ZJH + 180 C1） | **180**（仅 C1 fake） |
| 外部测试 | C2 **158** | **158**（一致） |
| ZJH val 20 例 / 183 图 | 从 ZJH 训练队列留出 | **未使用 ZJH** |

---

## 2. 数据划分

| 角色 | 中心 | 图像数 | Manifest | 是否参与训练 |
|------|------|--------|----------|--------------|
| 训练队列 | LymphUs fake **Center1** | 180 | `manifests/lymphus_center1_train.json` | 是 |
| 5 折 CV 池 | 同上（= 训练队列） | 180 | 同上 + `image_roots.json` | 折内 train/val |
| 外部测试 | LymphUs fake **Center2** | 158 | `manifests/lymphus_center2_test.json` | **否**（仅评估） |

**患者 ID（LymphUs fake）**：同一病例号的 Benign / Malignant 为不同图像、标签可不同；评估时按 **每张图一个患者簇**（`source` + 文件名），与 5 折 CV 分组一致。

---

## 3. 模型与训练配置（LLNM-Net）

| 项目 | 设置 |
|------|------|
| 模型 | `LLNM_Net`，`image_size=224`，`resize_size=256` |
| 优化 | AdamW，`lr=1e-5`，`weight_decay=0.01` |
| Batch / Epoch | `batch_size=4`，`epochs=50` |
| 选模 | 外部 C2 **图像级 AUROC** 最大 → `best_model.pth` |
| 迁移学习 | 未使用 `--pretrained_weights` |
| 数据增强 | 无（Resize + CenterCrop） |
| 临床/组学输入 | 默认 age/sex/report；**未**接入 `radiomics_csv` / `patient_info`（shape/echo 为 0，归一化 std≈1e-6） |
| CV 随机种子 | `cv_seed=42`，`n_splits=5` |

---

## 4. 患者级 5 折交叉验证（R2.11）

**输入**：Center1 全部 180 张；按 `patient_id` 分层 5 折（本数据 180 簇 = 180 图）。

### 4.1 各折验证集患者级 AUROC

| Fold | Val 图像数 | Val 患者簇数 | Val 患者级 AUROC |
|------|------------|--------------|------------------|
| 0 | 36 | 36 | 0.916 |
| 1 | 36 | 36 | 0.819 |
| 2 | 36 | 36 | 0.875 |
| 3 | 36 | 36 | 0.806 |
| 4 | 36 | 36 | 0.842 |

### 4.2 合并 OOF（180 张）

| 级别 | AUROC | AUPRC | Acc | 95% CI（AUROC，聚类 bootstrap） |
|------|-------|-------|-----|----------------------------------|
| 图像级 | 0.796 | 0.728 | 0.722 | [0.725, 0.860] |
| 患者级 | 0.796 | 0.728 | 0.722 | [0.727, 0.858] |

> fake 上图像级与患者级数值相同（一图一簇）。**OOF 仅在 Center1 内**，不能替代 Center2 外部性能。

**归档**：`new_code/runs/b9_fake_cv/cv_summary.json`、`oof_predictions.json`、各折 `fold_XX/fold_summary.json`。

---

## 5. 全量训练 + Center2 外部测试（R2.12）

**协议**：Center1 **180** 张全量训练 50 epoch；每个 epoch 在 Center2 **158** 张上评估；保存 C2 AUROC 最优权重。

| 指标 | 最优 checkpoint（**Epoch 29**） | 最后一轮（Epoch 50） |
|------|--------------------------------|----------------------|
| C2 图像级 AUROC | **0.683** | 0.591 |
| C2 患者级 AUROC | **0.683** | 0.591 |
| C2 Acc | — | 0.589 |
| 训练集 AUROC | — | 1.000（明显过拟合） |
| C2 AUROC 95% CI（患者级，epoch 29） | **[0.591, 0.767]** | [0.497, 0.681]（epoch 50） |

**返修建议引用**：外部 LymphUs Center2 以 **`best_model.pth`（epoch 29）** 为准，**不宜**引用 epoch 50。

| 汇总字段 | 值 |
|----------|-----|
| `summary.json` → `best_score` | 0.6826 |
| `train_samples` / `test_samples` | 180 / 158 |

**归档**：`new_code/runs/b9_fake_full_train/summary.json`、`history.json`、`test_outputs_epoch_*.json`。

---

## 6. 结论陈述（可直接改写入回复草稿）

1. **方法**：在 LNM 任务上采用 **LLNM-Net**；按审稿意见以 **患者级分层 5 折 CV** 在训练队列（本实验为 LymphUs Center1 fake，*n*=180）上报告 **OOF AUROC** 及 **聚类 bootstrap 95% CI**；外部泛化在 **未参与训练的 Center2**（*n*=158）上以 **患者级平均概率** 计算 AUROC 及 CI。  
2. **内部 CV（Center1 OOF）**：患者级 AUROC **0.796**（95% CI **0.727–0.858**）。  
3. **外部测试（Center2）**：最优模型患者级 AUROC **0.683**（95% CI **0.591–0.767**）。  
4. **过拟合**：小样本训练下训练 AUROC 可达 1.0，外部指标以 **验证 AUROC 选中的 checkpoint** 报告。  
5. **输入说明**：本跑通实验未启用放射组学/体格表；与主文全模态设定不一致时，需在 Methods 中写明 **本表为 fake 队列、影像为主的最小配置**。

---

## 7. 后续若恢复稿内完整训练队列

| 步骤 | Manifest | 输出目录（建议） |
|------|----------|------------------|
| 5 折 CV | `train_combined_zjh_plus_lymphus_c1.json` | `new_code/runs/b9_paper_cv` |
| 全量训练 + C2 | 同上 + `lymphus_center2_test.json` | `new_code/runs/b9_paper_full_train` |

替换本节 4–5 表格中的样本量与 AUROC；ZJH 患者为多图/患者，患者级与图像级 OOF **将不再相同**。

---

## 8. 复现命令

```bash
cd /mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net
export CUDA_VISIBLE_DEVICES=0
bash new_code/run_b9_lnm_fake.sh   # 已含 CV + 全量训练；评估指标已写入上述 runs
```

单步训练/评估见 `docs/B9_LNM_fake跑通.md`。
