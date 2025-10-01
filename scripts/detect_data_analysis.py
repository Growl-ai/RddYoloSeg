# detect_data_analysis.py
import os
from pathlib import Path
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 类别名称映射字典
CLASS_NAMES = {
    0: 'Longitudinal Crack',
    1: 'Transverse Crack',
    2: 'Aligator Crack',
    3: 'Pothole'
}

def count_classes_in_split(split_dir):
    """
    统计指定划分目录中的类别分布
    
    Args:
        split_dir: 划分目录路径，包含labels子目录
    """
    labels_dir = Path(split_dir)
    if not labels_dir.exists():
        print(f"错误: 找不到标签目录 {labels_dir}")
        return None
    
    # 统计每个类别的实例数量
    class_counts = defaultdict(int)
    total_instances = 0
    total_images = 0
    
    # 遍历所有标签文件
    for label_file in labels_dir.glob("*.txt"):
        total_images += 1
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip():  # 非空行
                try:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_instances += 1
                except (ValueError, IndexError):
                    # 忽略格式错误的行
                    continue
    
    return {
        "class_counts": dict(class_counts),
        "total_instances": total_instances,
        "total_images": total_images
    }

def analyze_dataset_distribution(data_dir):
    """
    分析数据集划分后的类别分布
    
    Args:
        data_dir: 数据集根目录，包含train、val、test子目录
    """
    splits = ["train", "val", "test"]
    results = {}
    
    for split in splits:
        split_dir = Path(data_dir) / "labels" / split
        if split_dir.exists():
            print(f"\n分析 {split} 集...")
            results[split] = count_classes_in_split(split_dir)
        else:
            print(f"警告: {split} 目录不存在: {split_dir}")
    
    return results

def print_statistics(results):
    """
    打印统计结果
    
    Args:
        results: 统计结果字典
    """
    # 获取所有类别
    all_classes = set()
    for split_data in results.values():
        if split_data:
            all_classes.update(split_data["class_counts"].keys())
    
    all_classes = sorted(all_classes)
    
    print("\n" + "="*100)
    print("数据集类别分布统计")
    print("="*100)
    
    # 打印表头
    header = f"{'类别ID':<8} {'类别名称':<20}"
    for split in results.keys():
        header += f"{split.capitalize():<15}"
    header += f"{'总计':<10}"
    print(header)
    print("-" * 100)
    
    # 打印每个类别的数量
    total_by_split = defaultdict(int)
    for class_id in all_classes:
        class_name = CLASS_NAMES.get(class_id, f"未知类别({class_id})")
        row = f"{class_id:<8} {class_name:<20}"
        class_total = 0
        for split, data in results.items():
            if data and class_id in data["class_counts"]:
                count = data["class_counts"][class_id]
                row += f"{count:<15}"
                class_total += count
                total_by_split[split] += count
            else:
                row += f"{0:<15}"
        row += f"{class_total:<10}"
        print(row)
    
    # 打印总计行
    print("-" * 100)
    total_row = f"{'总计':<8} {'':<20}"
    grand_total = 0
    for split in results.keys():
        if split in total_by_split:
            total_row += f"{total_by_split[split]:<15}"
            grand_total += total_by_split[split]
        else:
            total_row += f"{0:<15}"
    total_row += f"{grand_total:<10}"
    print(total_row)
    
    # 打印图像数量
    print("-" * 100)
    images_row = f"{'图像数':<8} {'':<20}"
    for split, data in results.items():
        if data:
            images_row += f"{data['total_images']:<15}"
        else:
            images_row += f"{0:<15}"
    print(images_row)
    print("="*100)

def plot_distribution(results, output_dir):
    """
    绘制类别分布图表
    
    Args:
        results: 统计结果字典
        output_dir: 输出目录
    """
    # 获取所有类别
    all_classes = set()
    for split_data in results.values():
        if split_data:
            all_classes.update(split_data["class_counts"].keys())
    
    all_classes = sorted(all_classes)
    
    # 准备数据
    splits = list(results.keys())
    class_counts = {split: [] for split in splits}
    
    for class_id in all_classes:
        for split in splits:
            if results[split] and class_id in results[split]["class_counts"]:
                class_counts[split].append(results[split]["class_counts"][class_id])
            else:
                class_counts[split].append(0)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 柱状图 - 各划分中各类别的数量
    x = np.arange(len(all_classes))
    width = 0.25
    multiplier = 0
    
    for split, counts in class_counts.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, counts, width, label=split.capitalize())
        multiplier += 1
    
    # 设置类别名称
    class_names = [CLASS_NAMES.get(class_id, f"未知({class_id})") for class_id in all_classes]
    
    ax1.set_xlabel('类别')
    ax1.set_ylabel('实例数量')
    ax1.set_title('各划分中各类别的实例数量')
    ax1.set_xticks(x + width, class_names, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    
    # 饼图 - 各划分的比例
    split_totals = []
    split_labels = []
    for split in splits:
        if results[split]:
            split_totals.append(results[split]["total_instances"])
            split_labels.append(f"{split.capitalize()}\n({results[split]['total_instances']} 实例)")
    
    ax2.pie(split_totals, labels=split_labels, autopct='%1.1f%%')
    ax2.set_title('数据集划分比例')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir) / "dataset_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n分布图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='统计YOLO数据集划分后的类别分布')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据集根目录，包含train、val、test子目录')
    parser.add_argument('--plot', action='store_true',
                       help='生成并保存分布图表')
    
    args = parser.parse_args()
    
    # 分析数据集分布
    results = analyze_dataset_distribution(args.data_dir)
    
    # 打印统计结果
    print_statistics(results)
    
    # 生成图表（如果启用）
    if args.plot:
        plot_distribution(results, args.data_dir)

if __name__ == "__main__":
    main()