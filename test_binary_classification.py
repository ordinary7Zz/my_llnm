"""
二分类性能测试脚本
用于评估LLNM-Net模型在二分类任务上的性能

使用方法（方式1 - 直接从图像文件夹加载，推荐）:
    python test_binary_classification.py --model_path <pth文件路径> --class0_dir <类别0图像路径> --class1_dir <类别1图像路径> --image_dir <图像根目录> --norm_params_file <归一化参数文件>

使用方法（方式2 - 使用pkl文件）:
    先运行: python prepare_data.py --mode folder --image_dir <图像根目录> --output test.pkl --label_mapping "class0:0,class1:1"
    再运行: python test_binary_classification.py --model_path <pth文件路径> --pkl_file test --image_dir <图像根目录>

示例:
    # 如果训练时使用了归一化（推荐）
    python test_binary_classification.py \
        --model_path model_epoch10.pth \
        --class0_dir dataset/test_NonMeta \
        --class1_dir dataset/test_Meta \
        --image_dir dataset \
        --norm_params_file train_norm_params.pkl
    
    # 如果训练时未使用归一化
    python test_binary_classification.py \
        --model_path model_epoch10.pth \
        --class0_dir dataset/test_NonMeta \
        --class1_dir dataset/test_Meta \
        --image_dir dataset

注意: 如果训练时使用了归一化（prepare_data.py的--normalize选项，默认启用），
     测试时必须提供 --norm_params_file 参数，否则性能会严重下降！
"""

from __future__ import print_function, division
import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
import argparse
from tqdm import tqdm
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from models.modeling_LLNM_Net import LLNM_Net, CONFIGS

tk_lim = 300  # report limit (必须与configs.py中的rr_len一致)


def load_weights(model, weight_path):
    """加载模型权重"""
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()
    
    # 处理DataParallel保存的权重（如果权重文件是用DataParallel保存的）
    if any(k.startswith('module.') for k in pretrained_weights.keys()):
        # 如果权重文件包含'module.'前缀，需要移除
        pretrained_weights = {k.replace('module.', ''): v for k, v in pretrained_weights.items()}
    
    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print(f"成功加载模型权重: {weight_path}")
    return model


class BinaryClassificationDataset(Dataset):
    """二分类数据集类 - 直接从图像文件夹加载"""
    def __init__(self, class0_dir, class1_dir, image_dir, transform=None, 
                 default_report="", default_age=50.0, default_sex=1.0,
                 default_img_feature=None, norm_params_file=None):
        """
        参数:
            class0_dir: 类别0的图像文件夹路径（相对于image_dir）
            class1_dir: 类别1的图像文件夹路径（相对于image_dir）
            image_dir: 图像根目录
            transform: 图像变换
            default_report: 默认文本报告
            default_age: 默认年龄
            default_sex: 默认性别（0=女，1=男）
            default_img_feature: 默认图像特征 [shape, echo]，如果为None则使用[0.0, 0.0]
            norm_params_file: 归一化参数文件路径（例如：train_norm_params.pkl）
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.default_report = default_report
        
        # 加载归一化参数（如果提供）
        self.norm_params = None
        if norm_params_file and os.path.exists(norm_params_file):
            try:
                with open(norm_params_file, 'rb') as f:
                    self.norm_params = pickle.load(f)
                
                # 检查必需的键是否存在
                required_keys = ['age_mean', 'age_std', 'sex_mean', 'sex_std', 
                                'shape_mean', 'shape_std', 'echo_mean', 'echo_std']
                missing_keys = [key for key in required_keys if key not in self.norm_params]
                
                if missing_keys:
                    print(f"\n{'='*70}")
                    print(f"❌ 错误: 归一化参数文件格式不正确!")
                    print(f"{'='*70}")
                    print(f"文件: {norm_params_file}")
                    print(f"缺少的键: {missing_keys}")
                    print(f"文件中的键: {list(self.norm_params.keys())}")
                    print()
                    print("可能原因:")
                    print("1. 归一化参数文件是旧版本生成的")
                    print("2. 归一化参数文件损坏")
                    print()
                    print("解决方案:")
                    print("重新生成训练数据和归一化参数文件:")
                    print("  python prepare_data.py \\")
                    print("    --mode folder \\")
                    print("    --image_dir ./dataset_processed/train \\")
                    print("    --output train.pkl \\")
                    print("    --label_mapping \"train_NonMeta:0,train_Meta:1\" \\")
                    print("    --patient_info \"./output/体格指标数据.xlsx\" \\")
                    print("    --radiomics_csv \"./output/radiomics_features.csv\"")
                    print(f"{'='*70}\n")
                    raise ValueError(f"归一化参数文件格式错误，缺少键: {missing_keys}")
                
                print(f"✓ 加载归一化参数: {norm_params_file}")
                print(f"  年龄: mean={self.norm_params['age_mean']:.2f}, std={self.norm_params['age_std']:.2f}")
                print(f"  性别: mean={self.norm_params['sex_mean']:.2f}, std={self.norm_params['sex_std']:.2f}")
                print(f"  Shape: mean={self.norm_params['shape_mean']:.6f}, std={self.norm_params['shape_std']:.6f}")
                print(f"  Echo: mean={self.norm_params['echo_mean']:.2f}, std={self.norm_params['echo_std']:.2f}")
                
                # 应用归一化到默认值
                self.default_age = (default_age - self.norm_params['age_mean']) / self.norm_params['age_std']
                self.default_sex = (default_sex - self.norm_params['sex_mean']) / self.norm_params['sex_std']
                if default_img_feature is None:
                    default_img_feature = [0.0, 0.0]
                self.default_img_feature = [
                    (default_img_feature[0] - self.norm_params['shape_mean']) / self.norm_params['shape_std'],
                    (default_img_feature[1] - self.norm_params['echo_mean']) / self.norm_params['echo_std']
                ]
                print(f"✓ 默认值（归一化后）:")
                print(f"  年龄: {default_age} -> {self.default_age:.4f}")
                print(f"  性别: {default_sex} -> {self.default_sex:.4f}")
                print(f"  Shape: {default_img_feature[0]} -> {self.default_img_feature[0]:.4f}")
                print(f"  Echo: {default_img_feature[1]} -> {self.default_img_feature[1]:.4f}")
            
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"❌ 错误: 无法加载归一化参数文件!")
                print(f"{'='*70}")
                print(f"文件: {norm_params_file}")
                print(f"错误信息: {e}")
                print()
                print("请重新生成归一化参数文件。")
                print(f"{'='*70}\n")
                raise
        else:
            # 不归一化，使用原始值
            self.default_age = default_age
            self.default_sex = default_sex
            self.default_img_feature = default_img_feature if default_img_feature is not None else [0.0, 0.0]
            if norm_params_file:
                print(f"警告: 未找到归一化参数文件: {norm_params_file}")
                print("将使用未归一化的默认值，这可能导致性能下降！")
            else:
                print("\n" + "="*70)
                print("⚠️  警告: 未提供归一化参数文件 (--norm_params_file)!")
                print("="*70)
                print("如果训练时使用了归一化（prepare_data.py 默认启用），")
                print("测试时必须提供归一化参数文件，否则性能会严重下降！")
                print()
                print("正确的命令示例:")
                print("  python test_binary_classification.py \\")
                print("    --model_path <模型路径> \\")
                print("    --class0_dir test_NonMeta \\")
                print("    --class1_dir test_Meta \\")
                print("    --image_dir ./dataset_processed/test \\")
                print("    --norm_params_file train_norm_params.pkl  # 添加此参数！")
                print()
                print("如需查找归一化参数文件，运行: ls *norm_params.pkl")
                print("="*70 + "\n")
        
        # 加载BERT模型和tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model = BertModel.from_pretrained('bert-base-chinese')
            self.bert_model.eval()  # 设置为评估模式
        except Exception as e:
            print(f"从Hugging Face加载BERT模型失败: {e}")
            print("请尝试以下解决方案:")
            print("1. 设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
            print("2. 或手动下载模型到本地后使用本地路径")
            raise
        
        # 收集图像路径和标签
        self.image_paths = []
        self.labels = []
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # 加载类别0的图像
        class0_full_path = self.image_dir / class0_dir
        if class0_full_path.exists():
            for img_path in class0_full_path.rglob('*'):
                if img_path.suffix.lower() in image_extensions:
                    # 保存相对路径（相对于image_dir）
                    rel_path = img_path.relative_to(self.image_dir)
                    self.image_paths.append(str(rel_path).replace('\\', '/'))
                    self.labels.append(0)
        else:
            print(f"警告: 类别0文件夹不存在: {class0_full_path}")
        
        # 加载类别1的图像
        class1_full_path = self.image_dir / class1_dir
        if class1_full_path.exists():
            for img_path in class1_full_path.rglob('*'):
                if img_path.suffix.lower() in image_extensions:
                    # 保存相对路径（相对于image_dir）
                    rel_path = img_path.relative_to(self.image_dir)
                    self.image_paths.append(str(rel_path).replace('\\', '/'))
                    self.labels.append(1)
        else:
            print(f"警告: 类别1文件夹不存在: {class1_full_path}")
        
        if len(self.image_paths) == 0:
            raise ValueError("未找到任何图像文件！请检查图像路径是否正确。")
        
        print(f"加载了 {len(self.image_paths)} 张图像")
        print(f"  类别0: {sum(1 for l in self.labels if l == 0)} 张")
        print(f"  类别1: {sum(1 for l in self.labels if l == 1)} 张")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_rel_path = self.image_paths[idx]
        img_full_path = self.image_dir / img_rel_path
        try:
            img = Image.open(img_full_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_full_path}: {e}")
            # 创建一个黑色图像作为占位符
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # 处理文本报告
        str_rr = self.default_report
        with torch.no_grad():
            input_ids = self.tokenizer.encode(str_rr, add_special_tokens=True, return_tensors='pt')
            outputs = self.bert_model(input_ids)
            last_hidden_state = outputs.last_hidden_state
            
            padding_length = tk_lim - last_hidden_state.shape[1]
            if padding_length > 0:
                padding_token = self.tokenizer.pad_token_id
                padding_tensor = torch.full((1, padding_length, last_hidden_state.shape[2]), padding_token)
                padded_outputs = torch.cat([last_hidden_state, padding_tensor], dim=1)
            else:
                padded_outputs = last_hidden_state
            
            rr_vector = padded_outputs[:, :tk_lim, :]
        
        rr = torch.tensor(rr_vector, dtype=torch.float32)
        demo = torch.tensor([self.default_age, self.default_sex], dtype=torch.float32)
        img_fea = torch.tensor(self.default_img_feature, dtype=torch.float32)
        
        return img, label, rr, demo, img_fea


class PklDataset(Dataset):
    """使用pkl文件的数据集类 - 与LLNM_Net.py中的Data类兼容"""
    def __init__(self, pkl_file, img_dir, transform=None, target_transform=None):
        dict_path = pkl_file + '.pkl'
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())  
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # 加载BERT模型和tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model = BertModel.from_pretrained('bert-base-chinese')
            self.bert_model.eval()
        except Exception as e:
            print(f"从Hugging Face下载失败: {e}")
            print("请尝试以下解决方案:")
            print("1. 设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
            print("2. 或手动下载模型到本地后使用本地路径")
            raise

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        k = self.idx_list[idx]
        img_path = os.path.join(self.img_dir, self.mm_data[k]['image'])
        img = Image.open(img_path).convert('RGB')

        label = self.mm_data[k]['label'].astype('float32')
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        str_rr = self.mm_data[k]['report']
        input_ids = self.tokenizer.encode(str_rr, add_special_tokens=True, return_tensors='pt')
        outputs = self.bert_model(input_ids)
        last_hidden_state = outputs.last_hidden_state

        padding_length = tk_lim - last_hidden_state.shape[1]
        if padding_length>0:
            padding_token = self.tokenizer.pad_token_id
            padding_tensor = torch.full((1, padding_length, last_hidden_state.shape[2]), padding_token)
            padded_outputs = torch.cat([last_hidden_state, padding_tensor], dim=1)
        else:
            padded_outputs = last_hidden_state
        
        rr_vector = padded_outputs[:, :tk_lim, :] 

        rr = torch.tensor(rr_vector, dtype=torch.float32)
        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()
        img_fea = torch.from_numpy(self.mm_data[k]['bts']).float()
        return img, label, rr, demo, img_fea


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    计算分类指标
    
    参数:
        y_true: 真实标签 (numpy array)
        y_pred_proba: 预测概率 (numpy array, shape: [n_samples, n_classes] 或 [n_samples])
        threshold: 分类阈值（默认0.5）
    
    返回:
        包含所有指标的字典
    """
    # 处理概率输入
    if y_pred_proba.ndim == 2:
        # 如果是二分类，使用正类（类别1）的概率
        if y_pred_proba.shape[1] == 2:
            y_pred_proba_class1 = y_pred_proba[:, 1]
        else:
            y_pred_proba_class1 = y_pred_proba[:, 0]
    else:
        y_pred_proba_class1 = y_pred_proba
    
    # 二值化预测
    y_pred = (y_pred_proba_class1 >= threshold).astype(int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算各项指标
    metrics = {}
    
    # AUROC (Area Under ROC Curve)
    try:
        metrics['AUROC'] = roc_auc_score(y_true, y_pred_proba_class1)
    except ValueError:
        metrics['AUROC'] = 0.5  # 如果所有标签都是同一类，返回0.5
    
    # AUPRC (Area Under Precision-Recall Curve)
    try:
        metrics['AUPRC'] = average_precision_score(y_true, y_pred_proba_class1)
    except ValueError:
        metrics['AUPRC'] = 0.0
    
    # Accuracy
    metrics['Acc'] = accuracy_score(y_true, y_pred)
    
    # Precision
    metrics['Prec'] = precision_score(y_true, y_pred, zero_division=0)
    
    # Recall (Sensitivity)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['Sensitivity'] = metrics['Recall']  # Recall和Sensitivity相同
    
    # F1 Score
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def test_model(model_path, dataset, batch_size=2, num_classes=2, device='cuda'):
    """
    测试模型性能
    
    参数:
        model_path: 模型权重文件路径 (.pth)
        dataset: 数据集对象
        batch_size: 批次大小（默认2，与训练时一致）
        num_classes: 类别数量（默认2）
        device: 设备 ('cuda' 或 'cpu')
    """
    # 设置设备
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    
    device = torch.device(device)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 创建模型
    config = CONFIGS["LLNM_Net"]
    model = LLNM_Net(config, 224, zero_head=True, num_classes=num_classes)
    
    # 加载权重
    model = load_weights(model, model_path)
    model = model.to(device)
    model.eval()
    
    # 进行预测
    print("\n开始测试...")
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels, rr, demo, img_fea in tqdm(dataloader, desc="测试进度"):
            # 移动到设备
            imgs = imgs.to(device)
            labels = labels.to(device)
            rr = rr.view(-1, tk_lim, rr.shape[3]).to(device).float()
            demo = demo.view(-1, 1, demo.shape[1]).to(device).float()
            img_fea = img_fea.view(-1, img_fea.shape[1], 1).to(device).float()
            sex = demo[:, :, 1].view(-1, 1, 1).to(device).float()
            age = demo[:, :, 0].view(-1, 1, 1).to(device).float()
            
            # 前向传播
            outputs = model(imgs, rr, img_fea, sex, age)
            logits = outputs[0]  # [batch_size, num_classes]
            
            # 计算概率
            probs = torch.sigmoid(logits)  # 使用sigmoid
            
            # 保存结果
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 合并所有结果
    y_true = np.concatenate(all_labels)
    y_pred_proba = np.concatenate(all_probs)
    
    # 处理标签维度
    if y_true.ndim > 1:
        y_true = y_true.squeeze()
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred_proba)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='LLNM-Net二分类性能测试')
    
    # 数据输入方式（二选一）
    parser.add_argument('--class0_dir', type=str, default=None,
                       help='类别0的图像文件夹路径（相对于image_dir），与--class1_dir一起使用（方式1）')
    parser.add_argument('--class1_dir', type=str, default=None,
                       help='类别1的图像文件夹路径（相对于image_dir），与--class0_dir一起使用（方式1）')
    parser.add_argument('--pkl_file', type=str, default=None,
                       help='pkl文件名（不含.pkl后缀），与--image_dir一起使用（方式2）')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重文件路径 (.pth)')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像根目录路径')
    
    # 可选参数
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批次大小 (默认: 2，与训练时一致)')
    parser.add_argument('--default_report', type=str, default='',
                       help='默认文本报告（如果图像没有对应的报告）')
    parser.add_argument('--default_age', type=float, default=50.0,
                       help='默认年龄 (默认: 50.0)')
    parser.add_argument('--default_sex', type=float, default=1.0,
                       help='默认性别，0=女，1=男 (默认: 1.0)')
    parser.add_argument('--default_img_feature', type=str, default=None,
                       help='默认图像特征 [shape, echo]，格式: "0.0,0.0" (默认: 0.0,0.0)')
    parser.add_argument('--norm_params_file', type=str, default=None,
                       help='归一化参数文件路径（例如：train_norm_params.pkl）。如果训练时使用了归一化，必须提供此参数！')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='类别数量 (默认: 2)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='使用的设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.class0_dir is None and args.class1_dir is None and args.pkl_file is None:
        raise ValueError("必须提供 --class0_dir 和 --class1_dir（方式1）或 --pkl_file（方式2）")
    
    if (args.class0_dir is not None or args.class1_dir is not None) and args.pkl_file is not None:
        raise ValueError("不能同时使用方式1（--class0_dir/--class1_dir）和方式2（--pkl_file），请选择其中一种")
    
    if args.class0_dir is not None and args.class1_dir is None:
        raise ValueError("必须同时提供 --class0_dir 和 --class1_dir")
    
    if args.class1_dir is not None and args.class0_dir is None:
        raise ValueError("必须同时提供 --class0_dir 和 --class1_dir")
    
    # 解析图像特征
    default_img_feature = None
    if args.default_img_feature:
        try:
            default_img_feature = [float(x.strip()) for x in args.default_img_feature.split(',')]
            if len(default_img_feature) != 2:
                raise ValueError("图像特征必须是两个值")
        except Exception as e:
            print(f"错误: 无法解析图像特征 '{args.default_img_feature}': {e}")
            print("使用默认值 [0.0, 0.0]")
            default_img_feature = [0.0, 0.0]
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 创建数据集
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    if args.pkl_file:
        # 方式2: 使用pkl文件
        print(f"使用pkl文件: {args.pkl_file}.pkl")
        dataset = PklDataset(args.pkl_file, args.image_dir, transform=data_transforms)
    else:
        # 方式1: 直接从图像文件夹加载
        print(f"从图像文件夹加载数据:")
        print(f"  类别0: {args.class0_dir}")
        print(f"  类别1: {args.class1_dir}")
        dataset = BinaryClassificationDataset(
            class0_dir=args.class0_dir,
            class1_dir=args.class1_dir,
            image_dir=args.image_dir,
            transform=data_transforms,
            default_report=args.default_report,
            default_age=args.default_age,
            default_sex=args.default_sex,
            default_img_feature=default_img_feature,
            norm_params_file=args.norm_params_file
        )
    
    # 运行测试
    metrics = test_model(
        model_path=args.model_path,
        dataset=dataset,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("二分类性能指标")
    print("="*60)
    print(f"AUROC (Area Under ROC Curve):        {metrics['AUROC']:.4f}")
    print(f"AUPRC (Area Under PR Curve):         {metrics['AUPRC']:.4f}")
    print(f"Accuracy:                            {metrics['Acc']:.4f}")
    print(f"Precision:                           {metrics['Prec']:.4f}")
    print(f"Recall:                              {metrics['Recall']:.4f}")
    print(f"F1 Score:                            {metrics['F1']:.4f}")
    print(f"Sensitivity:                         {metrics['Sensitivity']:.4f}")
    print(f"Specificity:                         {metrics['Specificity']:.4f}")
    print("="*60)
    
    # 保存结果到文件
    result_file = 'test_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("二分类性能指标\n")
        f.write("="*60 + "\n")
        f.write(f"模型路径: {args.model_path}\n")
        if args.pkl_file:
            f.write(f"数据文件: {args.pkl_file}.pkl\n")
            f.write(f"图像目录: {args.image_dir}\n")
        else:
            f.write(f"类别0路径: {args.class0_dir}\n")
            f.write(f"类别1路径: {args.class1_dir}\n")
            f.write(f"图像根目录: {args.image_dir}\n")
        f.write("="*60 + "\n")
        f.write(f"AUROC:        {metrics['AUROC']:.4f}\n")
        f.write(f"AUPRC:        {metrics['AUPRC']:.4f}\n")
        f.write(f"Accuracy:     {metrics['Acc']:.4f}\n")
        f.write(f"Precision:    {metrics['Prec']:.4f}\n")
        f.write(f"Recall:       {metrics['Recall']:.4f}\n")
        f.write(f"F1 Score:     {metrics['F1']:.4f}\n")
        f.write(f"Sensitivity:  {metrics['Sensitivity']:.4f}\n")
        f.write(f"Specificity:  {metrics['Specificity']:.4f}\n")
    
    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    main()


