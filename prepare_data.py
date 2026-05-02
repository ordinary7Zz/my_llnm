"""
数据准备脚本：将图像数据转换为LLNM-Net所需的pkl格式

使用方法：
1. 准备图像文件，按类别组织（可选）或使用CSV文件指定标签
2. 运行此脚本生成pkl文件
3. 使用生成的pkl文件训练模型
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import re

def extract_name_from_filename(filename):
    """
    从文件名中提取人名
    文件名格式: 严珍明__119a989b9c__严珍明_01_0007_0007.jpg
    返回: 严珍明
    """
    # 提取第一个双下划线之前的部分
    match = re.match(r'^([^_]+)__', filename)
    if match:
        return match.group(1)
    # 如果没有双下划线，尝试提取第一个下划线之前的部分
    match = re.match(r'^([^_]+)_', filename)
    if match:
        return match.group(1)
    # 如果都没有，返回文件名（不含扩展名）
    return os.path.splitext(filename)[0]

def create_pkl_from_folder(image_dir, output_pkl, label_mapping=None, default_report="", 
                          default_age=50, default_sex=1, patient_info_file=None, radiomics_csv=None,
                          shape_feature='original_shape2D_Elongation', echo_feature='original_firstorder_Mean',
                          normalize=True):
    """
    从图像文件夹创建pkl文件
    
    参数:
        image_dir: 图像文件夹路径（可以包含子文件夹，子文件夹名作为类别）
        output_pkl: 输出的pkl文件路径
        label_mapping: 类别名到标签的映射，例如 {'NonMeta': 0, 'Lateral': 1} 或 {'benign': 0, 'malignant': 1}
        default_report: 默认的文本报告（如果图像没有对应的报告）
        default_age: 默认年龄
        default_sex: 默认性别（0=女，1=男）
        patient_info_file: 体格指标数据Excel文件路径
        radiomics_csv: radiomics特征CSV文件路径
        shape_feature: 用作shape的radiomics特征名
        echo_feature: 用作echo的radiomics特征名
    """
    data_dict = {}
    image_dir = Path(image_dir)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 读取患者信息
    patient_info_dict = {}
    if patient_info_file and os.path.exists(patient_info_file):
        try:
            patient_df = pd.read_excel(patient_info_file)
            print(f"读取患者信息文件: {patient_info_file}")
            print(f"患者信息列: {patient_df.columns.tolist()}")
            # 创建姓名到年龄性别的映射
            for _, row in patient_df.iterrows():
                name = str(row['姓名']).strip() if '姓名' in row else None
                if name:
                    age = row['年龄'] if '年龄' in row and pd.notna(row['年龄']) else default_age
                    sex = row['性别'] if '性别' in row and pd.notna(row['性别']) else default_sex
                    # 性别转换：如果是文字，转为数字（女=0，男=1）
                    if isinstance(sex, str):
                        sex = 0 if sex in ['女', 'F', 'f', 'female', 'Female'] else 1
                    patient_info_dict[name] = {'age': float(age), 'sex': float(sex)}
            print(f"成功加载 {len(patient_info_dict)} 个患者信息")
            # 打印前5个患者姓名用于调试
            if patient_info_dict:
                sample_names = list(patient_info_dict.keys())[:5]
                print(f"患者姓名示例: {sample_names}")
        except Exception as e:
            print(f"警告: 读取患者信息文件失败: {e}")
    
    # 读取radiomics特征
    radiomics_dict = {}
    if radiomics_csv and os.path.exists(radiomics_csv):
        try:
            radiomics_df = pd.read_csv(radiomics_csv)
            print(f"读取radiomics文件: {radiomics_csv}")
            print(f"Radiomics特征数: {len(radiomics_df.columns)}")
            # 检查所需特征是否存在
            if shape_feature not in radiomics_df.columns:
                print(f"警告: shape特征 '{shape_feature}' 不存在，将使用默认值0.0")
            if echo_feature not in radiomics_df.columns:
                print(f"警告: echo特征 '{echo_feature}' 不存在，将使用默认值0.0")
            # 创建文件名到特征的映射
            for _, row in radiomics_df.iterrows():
                filename = str(row['filename']) if 'filename' in row else None
                if filename:
                    shape_val = row[shape_feature] if shape_feature in row and pd.notna(row[shape_feature]) else 0.0
                    echo_val = row[echo_feature] if echo_feature in row and pd.notna(row[echo_feature]) else 0.0
                    radiomics_dict[filename] = {'shape': float(shape_val), 'echo': float(echo_val)}
            print(f"成功加载 {len(radiomics_dict)} 个radiomics特征")
            # 打印前3个文件名示例用于调试
            if radiomics_dict:
                sample_files = list(radiomics_dict.keys())[:3]
                print(f"Radiomics文件名示例: {sample_files}")
        except Exception as e:
            print(f"警告: 读取radiomics文件失败: {e}")
    
    # 如果没有提供label_mapping，尝试从文件夹名推断
    if label_mapping is None:
        # 检查是否有子文件夹
        subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
        if subdirs:
            # 从子文件夹名创建映射
            unique_labels = sorted([d.name for d in subdirs])
            label_mapping = {name: idx for idx, name in enumerate(unique_labels)}
            print(f"自动检测到类别映射: {label_mapping}")
        else:
            # 所有图像使用同一标签（需要用户指定）
            print("警告: 未找到子文件夹，所有图像将使用标签0。请使用--label_mapping参数指定标签。")
            label_mapping = {}
    
    # 遍历图像文件
    image_count = 0
    for img_path in image_dir.rglob('*'):
        if img_path.suffix.lower() in image_extensions:
            # 获取相对路径（相对于image_dir）
            rel_path = img_path.relative_to(image_dir)
            
            # 确定标签
            if label_mapping:
                # 从父文件夹名获取标签
                parent_name = img_path.parent.name
                if parent_name in label_mapping:
                    label = label_mapping[parent_name]
                else:
                    # 如果父文件夹不在映射中，使用第一个标签
                    label = list(label_mapping.values())[0] if label_mapping else 0
            else:
                label = 0
            
            # 创建样本ID（使用文件名，不含扩展名）
            sample_id = f"sample_{image_count:04d}"
            
            # 从文件名提取人名
            filename_only = img_path.name
            patient_name = extract_name_from_filename(filename_only)
            
            # 获取年龄和性别
            age = default_age
            sex = default_sex
            matched = False
            if patient_name in patient_info_dict:
                age = patient_info_dict[patient_name]['age']
                sex = patient_info_dict[patient_name]['sex']
                matched = True
            
            # 调试信息：前5个样本打印匹配情况
            if image_count < 5:
                print(f"文件: {filename_only}")
                print(f"  提取姓名: '{patient_name}'")
                print(f"  匹配状态: {'✓ 成功' if matched else '✗ 失败(使用默认值)'}")
                print(f"  年龄={age}, 性别={sex}")
            
            # 获取shape和echo特征
            shape_val = 0.0
            echo_val = 0.0
            radiomics_matched = False
            if filename_only in radiomics_dict:
                shape_val = radiomics_dict[filename_only]['shape']
                echo_val = radiomics_dict[filename_only]['echo']
                radiomics_matched = True
            
            # 调试信息：前5个样本打印radiomics匹配情况
            if image_count < 5:
                print(f"  Radiomics匹配: {'✓ 成功' if radiomics_matched else '✗ 失败(使用默认值0.0)'}")
                print(f"  shape={shape_val:.6f}, echo={echo_val:.6f}")
            
            # 创建数据条目
            data_dict[sample_id] = {
                'image': str(rel_path).replace('\\', '/'),  # 使用正斜杠
                'label': np.array([label], dtype=np.float32),
                'report': default_report,
                'bics': np.array([age, sex], dtype=np.float32),  # [年龄, 性别]
                'bts': np.array([shape_val, echo_val], dtype=np.float32)  # [shape, echo]
            }
            image_count += 1
    
    # 数据归一化
    if normalize and image_count > 0:
        print("\n" + "="*50)
        print("开始数据归一化...")
        
        # 收集所有特征值
        ages = [v['bics'][0] for v in data_dict.values()]
        sexes = [v['bics'][1] for v in data_dict.values()]
        shapes = [v['bts'][0] for v in data_dict.values()]
        echos = [v['bts'][1] for v in data_dict.values()]
        
        # 计算统计量（性别不归一化）
        age_mean, age_std = np.mean(ages), np.std(ages)
        shape_mean, shape_std = np.mean(shapes), np.std(shapes)
        echo_mean, echo_std = np.mean(echos), np.std(echos)
        
        # 避免除以0
        age_std = max(age_std, 1e-6)
        shape_std = max(shape_std, 1e-6)
        echo_std = max(echo_std, 1e-6)
        
        print(f"归一化参数:")
        print(f"  年龄: mean={age_mean:.2f}, std={age_std:.2f}, 范围=[{min(ages):.2f}, {max(ages):.2f}]")
        print(f"  性别: 不归一化, 范围=[{min(sexes):.2f}, {max(sexes):.2f}]")
        print(f"  Shape: mean={shape_mean:.6f}, std={shape_std:.6f}, 范围=[{min(shapes):.6f}, {max(shapes):.6f}]")
        print(f"  Echo: mean={echo_mean:.2f}, std={echo_std:.2f}, 范围=[{min(echos):.2f}, {max(echos):.2f}]")
        
        # 应用归一化（性别不归一化）
        for sample_id in data_dict:
            data_dict[sample_id]['bics'][0] = (data_dict[sample_id]['bics'][0] - age_mean) / age_std
            # data_dict[sample_id]['bics'][1] 保持不变（性别不归一化）
            data_dict[sample_id]['bts'][0] = (data_dict[sample_id]['bts'][0] - shape_mean) / shape_std
            data_dict[sample_id]['bts'][1] = (data_dict[sample_id]['bts'][1] - echo_mean) / echo_std
        
        # 保存归一化参数（性别不包含归一化参数）
        norm_params = {
            'age_mean': age_mean, 'age_std': age_std,
            'shape_mean': shape_mean, 'shape_std': shape_std,
            'echo_mean': echo_mean, 'echo_std': echo_std
        }
        norm_params_file = output_pkl.replace('.pkl', '_norm_params.pkl')
        with open(norm_params_file, 'wb') as f:
            pickle.dump(norm_params, f)
        print(f"归一化参数已保存到: {norm_params_file}")
        print("="*50 + "\n")
    
    # 保存pkl文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"成功创建pkl文件: {output_pkl}")
    print(f"共处理 {image_count} 张图像")
    print(f"标签分布: {pd.Series([v['label'][0] for v in data_dict.values()]).value_counts().to_dict()}")
    
    return data_dict

def create_pkl_from_csv(csv_path, image_dir, output_pkl, 
                        image_col='image', label_col='label', 
                        report_col=None, age_col=None, sex_col=None,
                        default_report="", default_age=50, default_sex=1,
                        patient_info_file=None, radiomics_csv=None,
                        shape_feature='original_shape2D_Elongation', echo_feature='original_firstorder_Mean',
                        normalize=True):
    """
    从CSV文件创建pkl文件
    
    CSV文件应包含以下列（至少需要image和label）:
        - image: 图像文件名或相对路径
        - label: 标签（0或1）
        - report: 文本报告（可选）
        - age: 年龄（可选）
        - sex: 性别（可选，0=女，1=男）
    """
    df = pd.read_csv(csv_path)
    data_dict = {}
    image_dir = Path(image_dir)
    
    # 读取患者信息
    patient_info_dict = {}
    if patient_info_file and os.path.exists(patient_info_file):
        try:
            patient_df = pd.read_excel(patient_info_file)
            for _, row in patient_df.iterrows():
                name = str(row['姓名']).strip() if '姓名' in row else None
                if name:
                    age = row['年龄'] if '年龄' in row and pd.notna(row['年龄']) else default_age
                    sex = row['性别'] if '性别' in row and pd.notna(row['性别']) else default_sex
                    if isinstance(sex, str):
                        sex = 0 if sex in ['女', 'F', 'f', 'female', 'Female'] else 1
                    patient_info_dict[name] = {'age': float(age), 'sex': float(sex)}
            print(f"成功加载 {len(patient_info_dict)} 个患者信息")
            # 打印前5个患者姓名用于调试
            if patient_info_dict:
                sample_names = list(patient_info_dict.keys())[:5]
                print(f"患者姓名示例: {sample_names}")
        except Exception as e:
            print(f"警告: 读取患者信息文件失败: {e}")
    
    # 读取radiomics特征
    radiomics_dict = {}
    if radiomics_csv and os.path.exists(radiomics_csv):
        try:
            radiomics_df = pd.read_csv(radiomics_csv)
            for _, row in radiomics_df.iterrows():
                filename = str(row['filename']) if 'filename' in row else None
                if filename:
                    shape_val = row[shape_feature] if shape_feature in row and pd.notna(row[shape_feature]) else 0.0
                    echo_val = row[echo_feature] if echo_feature in row and pd.notna(row[echo_feature]) else 0.0
                    radiomics_dict[filename] = {'shape': float(shape_val), 'echo': float(echo_val)}
            print(f"成功加载 {len(radiomics_dict)} 个radiomics特征")
            # 打印前3个文件名示例用于调试
            if radiomics_dict:
                sample_files = list(radiomics_dict.keys())[:3]
                print(f"Radiomics文件名示例: {sample_files}")
        except Exception as e:
            print(f"警告: 读取radiomics文件失败: {e}")
    
    required_cols = [image_col, label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必需的列: {missing_cols}")
    
    for idx, row in df.iterrows():
        sample_id = f"sample_{idx:04d}"
        
        # 图像路径
        img_name = str(row[image_col])
        img_path = image_dir / img_name
        
        # 检查图像是否存在
        if not img_path.exists():
            print(f"警告: 图像文件不存在: {img_path}")
            continue
        
        # 标签
        label = float(row[label_col])
        
        # 文本报告
        if report_col and report_col in df.columns:
            report = str(row[report_col]) if pd.notna(row[report_col]) else default_report
        else:
            report = default_report
        
        # 从文件名提取人名
        patient_name = extract_name_from_filename(img_name)
        
        # 年龄 - 优先从CSV，其次从患者信息，最后用默认值
        matched = False
        if age_col and age_col in df.columns:
            age = float(row[age_col]) if pd.notna(row[age_col]) else default_age
        elif patient_name in patient_info_dict:
            age = patient_info_dict[patient_name]['age']
            matched = True
        else:
            age = default_age
        
        # 性别 - 优先从CSV，其次从患者信息，最后用默认值
        if sex_col and sex_col in df.columns:
            sex = float(row[sex_col]) if pd.notna(row[sex_col]) else default_sex
        elif patient_name in patient_info_dict:
            sex = patient_info_dict[patient_name]['sex']
            matched = True
        else:
            sex = default_sex
        
        # 调试信息：前5个样本打印匹配情况
        if idx < 5:
            print(f"文件: {img_name}")
            print(f"  提取姓名: '{patient_name}'")
            print(f"  匹配状态: {'✓ 成功' if matched else '✗ 失败(使用默认值)'}")
            print(f"  年龄={age}, 性别={sex}")
        
        # 获取shape和echo特征
        shape_val = 0.0
        echo_val = 0.0
        radiomics_matched = False
        if img_name in radiomics_dict:
            shape_val = radiomics_dict[img_name]['shape']
            echo_val = radiomics_dict[img_name]['echo']
            radiomics_matched = True
        
        # 调试信息：前5个样本打印radiomics匹配情况
        if idx < 5:
            print(f"  Radiomics匹配: {'✓ 成功' if radiomics_matched else '✗ 失败(使用默认值0.0)'}")
            print(f"  shape={shape_val:.6f}, echo={echo_val:.6f}")
        
        # 创建数据条目
        data_dict[sample_id] = {
            'image': img_name.replace('\\', '/'),
            'label': np.array([label], dtype=np.float32),
            'report': report,
            'bics': np.array([age, sex], dtype=np.float32),
            'bts': np.array([shape_val, echo_val], dtype=np.float32)
        }
    
    # 数据归一化
    if normalize and len(data_dict) > 0:
        print("\n" + "="*50)
        print("开始数据归一化...")
        
        # 收集所有特征值
        ages = [v['bics'][0] for v in data_dict.values()]
        sexes = [v['bics'][1] for v in data_dict.values()]
        shapes = [v['bts'][0] for v in data_dict.values()]
        echos = [v['bts'][1] for v in data_dict.values()]
        
        # 计算统计量（性别不归一化）
        age_mean, age_std = np.mean(ages), np.std(ages)
        shape_mean, shape_std = np.mean(shapes), np.std(shapes)
        echo_mean, echo_std = np.mean(echos), np.std(echos)
        
        # 避免除以0
        age_std = max(age_std, 1e-6)
        shape_std = max(shape_std, 1e-6)
        echo_std = max(echo_std, 1e-6)
        
        print(f"归一化参数:")
        print(f"  年龄: mean={age_mean:.2f}, std={age_std:.2f}, 范围=[{min(ages):.2f}, {max(ages):.2f}]")
        print(f"  性别: 不归一化, 范围=[{min(sexes):.2f}, {max(sexes):.2f}]")
        print(f"  Shape: mean={shape_mean:.6f}, std={shape_std:.6f}, 范围=[{min(shapes):.6f}, {max(shapes):.6f}]")
        print(f"  Echo: mean={echo_mean:.2f}, std={echo_std:.2f}, 范围=[{min(echos):.2f}, {max(echos):.2f}]")
        
        # 应用归一化（性别不归一化）
        for sample_id in data_dict:
            data_dict[sample_id]['bics'][0] = (data_dict[sample_id]['bics'][0] - age_mean) / age_std
            # data_dict[sample_id]['bics'][1] 保持不变（性别不归一化）
            data_dict[sample_id]['bts'][0] = (data_dict[sample_id]['bts'][0] - shape_mean) / shape_std
            data_dict[sample_id]['bts'][1] = (data_dict[sample_id]['bts'][1] - echo_mean) / echo_std
        
        # 保存归一化参数（性别不包含归一化参数）
        norm_params = {
            'age_mean': age_mean, 'age_std': age_std,
            'shape_mean': shape_mean, 'shape_std': shape_std,
            'echo_mean': echo_mean, 'echo_std': echo_std
        }
        norm_params_file = output_pkl.replace('.pkl', '_norm_params.pkl')
        with open(norm_params_file, 'wb') as f:
            pickle.dump(norm_params, f)
        print(f"归一化参数已保存到: {norm_params_file}")
        print("="*50 + "\n")
    
    # 
    # 保存pkl文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"成功创建pkl文件: {output_pkl}")
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='是否归一化bics和bts特征（默认：True）')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='禁用归一化')
    print(f"共处理 {len(data_dict)} 个样本")
    print(f"标签分布: {pd.Series([v['label'][0] for v in data_dict.values()]).value_counts().to_dict()}")
    
    return data_dict

def main():
    parser = argparse.ArgumentParser(description='准备LLNM-Net训练数据')
    parser.add_argument('--mode', type=str, choices=['folder', 'csv'], default='folder',
                       help='数据输入模式: folder（从文件夹）或 csv（从CSV文件）')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像文件夹路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出的pkl文件路径（例如: train.pkl）')
    
    # 文件夹模式参数
    parser.add_argument('--label_mapping', type=str, default=None,
                       help='类别映射，格式: "NonMeta:0,Lateral:1" 或 "benign:0,malignant:1"')
    
    # CSV模式参数
    parser.add_argument('--csv', type=str, default=None,
                       help='CSV文件路径（CSV模式必需）')
    parser.add_argument('--image_col', type=str, default='image',
                       help='CSV中图像列名')
    parser.add_argument('--label_col', type=str, default='label',
                       help='CSV中标签列名')
    parser.add_argument('--report_col', type=str, default=None,
                       help='CSV中报告列名（可选）')
    parser.add_argument('--age_col', type=str, default=None,
                       help='CSV中年龄列名（可选）')
    parser.add_argument('--sex_col', type=str, default=None,
                       help='CSV中性别列名（可选）')
    
    # 默认值参数
    parser.add_argument('--default_report', type=str, default='',
                       help='默认文本报告（当没有提供时使用）')
    parser.add_argument('--default_age', type=float, default=50.0,
                       help='默认年龄')
    parser.add_argument('--default_sex', type=int, default=1,
                       help='默认性别（0=女，1=男）')
    
    # 患者信息和radiomics参数
    parser.add_argument('--patient_info', type=str, default=None,
                       help='体格指标数据Excel文件路径（包含姓名、年龄、性别列）')
    parser.add_argument('--radiomics_csv', type=str, default=None,
                       help='Radiomics特征CSV文件路径（包含filename和特征列）')
    parser.add_argument('--shape_feature', type=str, default='original_shape2D_Elongation',
                       help='用作shape的radiomics特征名')
    parser.add_argument('--echo_feature', type=str, default='original_firstorder_Mean',
                       help='用作echo的radiomics特征名')
    
    args = parser.parse_args()
    
    # 解析label_mapping
    label_mapping = None
    if args.label_mapping:
        label_mapping = {}
        for pair in args.label_mapping.split(','):
            key, value = pair.split(':')
            label_mapping[key.strip()] = int(value.strip())
    
    if args.mode == 'folder':
        create_pkl_from_folder(
            args.image_dir,
            args.output,
            label_mapping=label_mapping,
            default_report=args.default_report,
            default_age=args.default_age,
            default_sex=args.default_sex,
            patient_info_file=args.patient_info,
            radiomics_csv=args.radiomics_csv,
            shape_feature=args.shape_feature,
            echo_feature=args.echo_feature
        )
    elif args.mode == 'csv':
        if not args.csv:
            raise ValueError("CSV模式需要提供--csv参数")
        create_pkl_from_csv(
            args.csv,
            args.image_dir,
            args.output,
            image_col=args.image_col,
            label_col=args.label_col,
            report_col=args.report_col,
            age_col=args.age_col,
            sex_col=args.sex_col,
            default_report=args.default_report,
            default_age=args.default_age,
            default_sex=args.default_sex,
            patient_info_file=args.patient_info,
            radiomics_csv=args.radiomics_csv,
            shape_feature=args.shape_feature,
            echo_feature=args.echo_feature,
            normalize=args.normalize
        )

if __name__ == '__main__':
    main()

