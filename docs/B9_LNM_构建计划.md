# B9 任务：LNM 在 LLNM-Net 上的构建计划

内部文档，对应 ThyroidXAgent 返修计划 **B9**（R2.11、R2.12；R2.10 为 FTC/PTC，不在本仓库 LNM 主线内）。

## 1. B9 要求摘要（仅 LNM）

| 审稿/返修项 | 要求 | 本仓库交付 |
|-------------|------|------------|
| R2.11 样本与过拟合 | 训练集规模写清；说明迁移/增强/正则；**20 例 val → 训练队列患者级 5 折 CV** | 脚本产出 CV 汇总 JSON + 可写入 Methods 的配置快照 |
| R2.12 标签与输入 | ZJH 原发灶 CN0/CN1 vs LymphUs 淋巴结图；术语统一；**患者级聚合（如平均概率）** | manifest 带来源字段；评估报告 image + patient 两级 |
| 数字 | 合并训练 **4936**（4756 ZJH + 180 C1）；外部 **C2 158** | 数据目录 `LNM_ThyroidXAgent_ZJH_LymphUs_fake`；`dataset_metadata.json` 记录与论文差异 |
| B2 交叉 | 置信区间为 **聚类（患者）bootstrap** | 与 B9 共用 `utils` 中患者级 bootstrap |

**不在本计划内（但 B9 条目提及）**：FTC/PTC 重训（F7）、ThyroidXAgent 的 DINOv3+AutoGluon 管线（本仓库为 LLNM-Net）。

## 2. 当前基线

| 模块 | 现状 | 缺口 |
|------|------|------|
| `new_code/train.py` | 双 JSON、单/双 `image_root`；图像级指标选 best | 无 5-fold；无训练配置落盘；合并 ZJH+C1 需多 root |
| `new_code/evaluate.py` | 图像级 AUROC + 图像 bootstrap | 无患者级聚合与聚类 bootstrap |
| `new_code/utils.py` | `build_sample_records` 隐式患者名 | 无 `patient_id` 字段；无多 root |
| `new_code/build_lnm_paper_dataset.py` | 论文式 manifest（val 20/183、C1/C2） | 已生成，见 `Classifaction_Data/LNM_ThyroidXAgent_ZJH_LymphUs_fake` |
| `new_code/dataset.py` | Resize/CenterCrop，无增强 | Methods 需与实现一致（默认不写增强） |

## 3. 数据约定（训练队列）

**论文逻辑**（本地路径见 `LNM_ThyroidXAgent_ZJH_LymphUs_fake/README.md`）：

1. **ZJH 恶性原发灶**：`Malignant_ultrasound_images_cropped`，标签 `LNM_CN01∈{0,1}`；患者级 **val 20 例 / 183 图**（固定种子，见 `dataset_metadata.json`）。
2. **LymphUs 训练补充**：fake 集 **Center1 180 张**（`lymphus_center1_train_LLNM_only.json`）。
3. **合并训练队列**：`train_combined_zjh_plus_lymphus_c1.json`（含 `image_root_key`: `zjh` | `lymphus_fake`）。
4. **外部测试**：**Center2 158 张**（不参与 CV 折内划分）。

**与稿内 4756/4936 差异**：本地 ZJH 标注约 8312 张；返修文稿需用 `actual_counts` 说明或后续再筛子集。

## 4. 代码构建阶段

### 阶段 A — 数据与评估基础（优先）

1. **`utils.py`**
   - `build_sample_records`：写入 `patient_id`（manifest 优先，否则文件名解析）。
   - 支持 `image_roots: dict[str, Path]` + 记录字段 `image_root_key`。
   - `aggregate_predictions_by_patient(probs, labels, patient_ids, strategy="mean")`。
   - `calculate_binary_metrics_confidence_intervals_clustered(...)`（按患者重采样）。

2. **`dataset.py` / `engine.py`**
   - 按样本解析图像路径（多 root）。
   - `evaluate` 输出含 `patient_ids`。

3. **`evaluate.py`**
   - 默认同时报告 `metrics_image` 与 `metrics_patient`；写出患者级预测表。

### 阶段 B — B9 核心：患者级 5 折 CV

4. **`new_code/run_b9_patient_cv.py`**
   - 输入：`train_combined` manifest + `image_roots.json`。
   - **按 `patient_id` 分组** 的 5-fold（分层依据患者多数标签或患者标签）。
   - 每折：训练折 fit `norm_stats`；验证折 OOF 概率；汇总 **OOF AUROC/AUPRC**（图像级 + 患者级）。
   - 输出：`runs/b9_cv/cv_summary.json`、`oof_predictions.json`、每折 `fold_k/`（可选 `best_model.pth`）。
   - 超参：复用 `train.py` 默认（epochs、lr、weight_decay）；折内早停可选（首版用固定 epoch + best val AUROC）。

5. **全量训练 + 外部测试（可选第二条命令）**
   - 在 **全部合并训练集**（不含 val 患者之 ZJH 部分 + C1）上训一版 final model；
   - `evaluate.py` 在 **C2** 上报告 **患者级** 指标（返修稿 LymphUs Center 2）。

### 阶段 C — 工程与文档

6. **`train.py`**
   - 训练结束写入 `train_config.json`（参数 + 样本数）。
   - 可选 `--image_roots_json` + 多 root manifest（与 CV 一致）。

7. **文档**
   - 更新 `new_code/my_run.md` 增加 B9 一节（命令示例）。
   - Methods 占位：迁移学习 = `--pretrained_weights`；增强 = 当前无（或后续 `--train_augment`）。

## 5. 推荐执行顺序（实验）

```text
1. python new_code/build_lnm_paper_dataset.py   # 若 manifest 未生成
2. python new_code/run_b9_patient_cv.py ...     # 5-fold OOF → Table S5 CV 行
3. python new_code/train.py ...                 # 全量合并训练（双 root）
4. python new_code/evaluate.py ...              # C2 外部，患者级 AUROC
```

## 6. 验收标准

- [ ] `run_b9_patient_cv.py` 可跑通合并训练 manifest，输出 5 折 OOF 与 **患者级** CV AUROC。
- [ ] `evaluate.py` 对外部 C2 输出 **患者级** AUROC 与 **聚类 bootstrap 95% CI**。
- [ ] manifest 中 ZJH / LymphUs 字段可追溯（`source`、`image_root_key`）。
- [ ] `train_config.json` / `cv_summary.json` 可供填 Methods 与 Table S5。

## 7. 风险与依赖

- **F7**：CN0/CN1 临床定义最终措辞在文稿侧；代码仅用 `LNM_CN01`。
- **fake vs 正式 LymphUs**：当前 manifest 指向 `Lymph_Node_Metastasis_fake`；换正式集只需重跑 dataset 构建并改 `image_roots.json`。
- **BERT 权重路径**：`dataset.py` 仍指向仓库内 `my_pretrained/`；环境缺失会导致训练失败（与 B9 无关但阻塞运行）。
- **算力**：5-fold × 50 epoch 成本较高；CV 脚本支持 `--epochs`、`--folds` 调试。

## 8. 实现状态跟踪

| 项 | 状态 |
|----|------|
| `build_lnm_paper_dataset.py` | 已完成 |
| 本计划文档 | 已完成 |
| `utils.py` 患者聚合 / 聚类 bootstrap / 多 `image_root` | 已完成 |
| `dataset.py` / `engine.py` / `evaluate.py` 患者级评估 | 已完成 |
| `run_b9_patient_cv.py` 5 折 CV | 已完成 |
| `train.py` `train_config.json` + `--image_roots_json` | 已完成 |
| `new_code/my_run.md` B9 命令 | 已完成 |
| 全量训练后自动外部 C2 一键脚本 | 未做（见 my_run 分步命令） |
