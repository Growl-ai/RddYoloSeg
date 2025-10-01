# resize_dataset.py 改名 resplit_seg_data.py
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml

def analyze_dataset_structure(data_dir):
    """
    分析数据集结构
    """
    data_dir = Path(data_dir)
    print(f"分析数据集结构: {data_dir}")
    
    # 检查目录结构
    sets = ['train', 'val', 'test']
    for set_name in sets:
        set_dir = data_dir / set_name
        if not set_dir.exists():
            print(f"警告: {set_name} 目录不存在")
            continue
            
        images_dir = set_dir / 'images'
        labels_dir = set_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"警告: {set_name} 目录结构不完整")
            continue
            
        image_files = list(images_dir.glob('*.*'))
        label_files = list(labels_dir.glob('*.txt'))
        
        print(f"{set_name}: {len(image_files)} 张图像, {len(label_files)} 个标签文件")
        
        # 分析类别分布
        class_counts = defaultdict(int)
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
        
        print(f"{set_name} 类别分布: {dict(class_counts)}")

def get_class_distribution(labels_dir):
    """
    获取标签目录中的类别分布
    """
    class_counts = defaultdict(int)
    label_files = list(labels_dir.glob('*.txt'))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    
    return class_counts, len(label_files)

def resize_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    重新划分数据集
    
    Args:
        data_dir: 数据集根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 检查比例总和是否为1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("训练集、验证集和测试集的比例之和必须为1")
    
    data_dir = Path(data_dir)
    
    # 创建临时目录用于收集所有数据
    all_data_dir = data_dir / 'all_data'
    all_images_dir = all_data_dir / 'images'
    all_labels_dir = all_data_dir / 'labels'
    
    # 如果已存在所有数据目录，则删除
    if all_data_dir.exists():
        shutil.rmtree(all_data_dir)
    
    # 创建目录
    all_images_dir.mkdir(parents=True, exist_ok=True)
    all_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有数据
    sets = ['train', 'val', 'test']
    for set_name in sets:
        set_dir = data_dir / set_name
        if not set_dir.exists():
            continue
            
        images_dir = set_dir / 'images'
        labels_dir = set_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        # 复制图像文件
        for img_file in images_dir.glob('*.*'):
            shutil.copy2(img_file, all_images_dir / img_file.name)
        
        # 复制标签文件
        for label_file in labels_dir.glob('*.txt'):
            shutil.copy2(label_file, all_labels_dir / label_file.name)
    
    # 获取所有图像和标签文件
    all_image_files = list(all_images_dir.glob('*.*'))
    all_label_files = list(all_labels_dir.glob('*.txt'))
    
    print(f"总共收集到 {len(all_image_files)} 张图像和 {len(all_label_files)} 个标签文件")
    
    # 获取类别分布
    class_counts, total_labels = get_class_distribution(all_labels_dir)
    print(f"总体类别分布: {dict(class_counts)}")
    
    # 按类别分组文件
    class_files = defaultdict(list)
    for label_file in all_label_files:
        with open(label_file, 'r') as f:
            classes_in_file = set()
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    classes_in_file.add(class_id)
            
            # 如果文件包含多个类别，将其分配给样本最少的类别
            if classes_in_file:
                # 找到样本最少的类别
                min_class = min(classes_in_file, key=lambda c: class_counts.get(c, 0))
                class_files[min_class].append((label_file, all_images_dir / (label_file.stem + '.jpg')))
    
    # 创建新的划分
    new_train_dir = data_dir / 'new_train'
    new_val_dir = data_dir / 'new_val'
    new_test_dir = data_dir / 'new_test'
    
    for dir_path in [new_train_dir, new_val_dir, new_test_dir]:
        images_dir = dir_path / 'images'
        labels_dir = dir_path / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 对每个类别进行分层抽样
    for class_id, files in class_files.items():
        random.shuffle(files)
        
        total_files = len(files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        # 分配训练集
        for label_file, img_file in files[:train_end]:
            if img_file.exists():
                shutil.copy2(img_file, new_train_dir / 'images' / img_file.name)
                shutil.copy2(label_file, new_train_dir / 'labels' / label_file.name)
        
        # 分配验证集
        for label_file, img_file in files[train_end:val_end]:
            if img_file.exists():
                shutil.copy2(img_file, new_val_dir / 'images' / img_file.name)
                shutil.copy2(label_file, new_val_dir / 'labels' / label_file.name)
        
        # 分配测试集
        for label_file, img_file in files[val_end:]:
            if img_file.exists():
                shutil.copy2(img_file, new_test_dir / 'images' / img_file.name)
                shutil.copy2(label_file, new_test_dir / 'labels' / label_file.name)
    
    # 分析新划分的类别分布
    print("\n新划分的类别分布:")
    for set_name, set_dir in [('训练集', new_train_dir), ('验证集', new_val_dir), ('测试集', new_test_dir)]:
        labels_dir = set_dir / 'labels'
        if labels_dir.exists():
            class_counts, total_files = get_class_distribution(labels_dir)
            print(f"{set_name}: {dict(class_counts)} (共 {total_files} 个样本)")
    
    # 清理临时目录
    shutil.rmtree(all_data_dir)
    
    print(f"\n重新划分完成! 新数据集保存在:")
    print(f"训练集: {new_train_dir}")
    print(f"验证集: {new_val_dir}")
    print(f"测试集: {new_test_dir}")

def update_data_yaml(data_dir, class_names):
    """
    更新数据配置文件
    """
    data_dir = Path(data_dir)
    yaml_path = data_dir / 'yolov8_seg.yaml'
    
    # 创建新的数据配置
    data_config = {
        'path': str(data_dir),
        'train': 'new_train/images',
        'val': 'new_val/images',
        'test': 'new_test/images',
        'names': class_names,
        'nc': len(class_names)
    }
    
    # 保存配置
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"\n数据配置文件已更新: {yaml_path}")

if __name__ == '__main__':
    # 数据集路径
    data_dir = "E:/RoadDefect/data/seg"
    
    # 类别名称映射 (根据您的实际类别调整)
    class_names = {
        0: "Crack-alligator",
        1: "Crack-long", 
        2: "Crack-trans",
        3: "Pothole"
    }
    
    # 首先分析当前数据集结构
    analyze_dataset_structure(data_dir)
    
    # 重新划分数据集 (70% 训练, 15% 验证, 15% 测试)
    resize_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # 更新数据配置文件
    update_data_yaml(data_dir, class_names)