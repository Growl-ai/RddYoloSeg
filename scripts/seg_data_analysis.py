import os
import numpy as np
import matplotlib.pyplot as plt

# 定义类别，确保与你的数据集一致
# 类别顺序和数量必须与 yolov8.yaml 文件中的 names 列表完全匹配
CLASSES = {
    0: 'Crack-alligator',
    1: 'Crack-long',
    2: 'Crack-trans',
    3: 'Pothole'
}

def analyze_dataset(base_dir):
    """
    遍历数据集目录，统计每个类别的实例数量。
    
    Args:
        base_dir (str): 数据集根目录。
    """
    train_counts = {class_name: 0 for class_name in CLASSES.values()}
    val_counts = {class_name: 0 for class_name in CLASSES.values()}
    test_counts = {class_name: 0 for class_name in CLASSES.values()}

    # 遍历 train, val, test 子集
    for split, counts_dict in zip(['train', 'val', 'test'], [train_counts, val_counts, test_counts]):
        labels_dir = os.path.join(base_dir, split, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"Warning: Labels directory for {split} not found. Skipping.")
            continue
            
        print(f"Analyzing {split} set...")
        
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                label_path = os.path.join(labels_dir, filename)
                try:
                    with open(label_path, 'r') as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id in CLASSES:
                                    counts_dict[CLASSES[class_id]] += 1
                except Exception as e:
                    print(f"Could not process {label_path}: {e}")
                    
    return train_counts, val_counts, test_counts

def plot_class_distribution(class_counts, title):
    """
    绘制并保存类别实例分布图。
    """
    if not class_counts or all(count == 0 for count in class_counts.values()):
        print(f"No data to plot for {title}.")
        return

    class_names = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts, color='skyblue')
    plt.title(title, fontsize=16)
    plt.xlabel('Defect Class', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 创建保存目录
    output_dir = './results/logs'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    filename = f"{title.replace(' ', '_')}_distribution.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

if __name__ == '__main__':
    # 将此路径修改为你的分割数据集根目录
    seg_dataset_path = 'data/seg'
    
    train_counts, val_counts, test_counts = analyze_dataset(seg_dataset_path)
    
    print("\nTraining set class counts:", train_counts)
    print("Validation set class counts:", val_counts)
    print("Test set class counts:", test_counts)
        
    plot_class_distribution(train_counts, 'Segmentation Training Set Class Distribution')
    plot_class_distribution(val_counts, 'Segmentation Validation Set Class Distribution')
    plot_class_distribution(test_counts, 'Segmentation Test Set Class Distribution')
    
    print("\nAnalysis complete. Plots saved in results/logs folder.")