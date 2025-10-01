import os
import random
import shutil
from pathlib import Path
import argparse
from collections import defaultdict
import datetime
def split_yolo_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    按类别比例划分YOLO格式的目标检测数据集
    
    Args:
        data_dir: 原始数据集目录，包含images和labels子目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 检查比例总和是否为1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("训练集、验证集和测试集的比例之和必须为1")
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    images_dir = Path(data_dir) / "images"
    labels_dir = Path(data_dir) / "labels"
    
    image_files = list(images_dir.glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    # 按类别组织图像文件
    class_images = defaultdict(list)
    
    for img_path in image_files:
        # 获取对应的标签文件
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"警告: 找不到标签文件 {label_path}")
            continue
            
        # 读取标签文件，获取类别信息
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # 提取所有类别
        classes_in_image = set()
        for line in lines:
            if line.strip():
                class_id = int(line.split()[0])
                classes_in_image.add(class_id)
                
        # 如果图像中没有目标，则跳过
        if not classes_in_image:
            print(f"警告: 图像 {img_path} 中没有检测目标")
            continue
            
        # 将图像添加到对应类别的列表中
        for class_id in classes_in_image:
            class_images[class_id].append((img_path, label_path))
    
    # 统计每个类别的图像数量
    print("类别分布统计:")
    for class_id, images in class_images.items():
        print(f"类别 {class_id}: {len(images)} 张图像")
    
    # 为每个类别划分数据集
    train_set = []
    val_set = []
    test_set = []
    
    for class_id, images in class_images.items():
        # 打乱当前类别的图像
        random.shuffle(images)
        
        # 计算每个划分的数量
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        # 划分数据集
        train_set.extend(images[:n_train])
        val_set.extend(images[n_train:n_train+n_val])
        test_set.extend(images[n_train+n_val:])
    
    # 去重（因为同一张图像可能包含多个类别）
    train_set = list(set(train_set))
    val_set = list(set(val_set))
    test_set = list(set(test_set))
    
    # 打乱各划分中的图像顺序
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    # 复制文件到对应目录
    def copy_files(file_list, split_name):
        for img_path, label_path in file_list:
            # 复制图像
            shutil.copy2(img_path, output_path / "images" / split_name / img_path.name)
            # 复制标签
            shutil.copy2(label_path, output_path / "labels" / split_name / label_path.name)
    
    print("\n正在复制文件...")
    copy_files(train_set, "train")
    copy_files(val_set, "val")
    copy_files(test_set, "test")
    
    # 统计最终划分结果
    print(f"\n划分完成:")
    print(f"训练集: {len(train_set)} 张图像")
    print(f"验证集: {len(val_set)} 张图像")
    print(f"测试集: {len(test_set)} 张图像")
    
    # 创建数据集配置文件
    create_dataset_yaml(output_path, len(class_images))
    
    return output_path

def create_dataset_yaml(output_path, num_classes):
    """创建YOLO数据集配置文件"""
    yaml_content = f"""# YOLO 数据集配置文件
# 生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# 数据集路径
path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

# 类别数量
nc: {num_classes}

# 类别名称
names: {list(range(num_classes))}
"""
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n数据集配置文件已创建: {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='按类别比例划分YOLO格式检测数据集')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='原始数据集目录，包含images和labels子目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 划分数据集
    output_path = split_yolo_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"\n数据集已成功划分到: {output_path}")