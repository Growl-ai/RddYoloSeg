import os
import shutil

def rename_dataset_files(base_dir):
    """
    遍历数据集目录，统一图片和标签的文件命名。
    
    Args:
        base_dir (str): 数据集根目录，例如 'E:/RoadDefect/dataset'。
    """
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(base_dir, split, 'images')
        labels_dir = os.path.join(base_dir, split, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Warning: Directory not found for split: {split}. Skipping.")
            continue
            
        print(f"Processing {split} set...")
        
        # 获取所有图片和标签文件的基础名称（不含扩展名）
        image_basenames = {os.path.splitext(f)[0] for f in os.listdir(images_dir)}
        label_basenames = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)}
        
        # 找到图片和标签都有的匹配项
        common_basenames = image_basenames.intersection(label_basenames)
        
        if not common_basenames:
            print(f"No matching image and label files found in {split} set. Skipping.")
            continue
            
        # 按字母排序，确保命名有序
        sorted_basenames = sorted(list(common_basenames))
        
        for i, basename in enumerate(sorted_basenames):
            new_filename_prefix = f"{split}_{i:05d}"
            
            # 重命名图片文件
            old_image_path = os.path.join(images_dir, f"{basename}.jpg")
            new_image_path = os.path.join(images_dir, f"{new_filename_prefix}.jpg")
            shutil.move(old_image_path, new_image_path)
            
            # 重命名标签文件
            old_label_path = os.path.join(labels_dir, f"{basename}.txt")
            new_label_path = os.path.join(labels_dir, f"{new_filename_prefix}.txt")
            shutil.move(old_label_path, new_label_path)
            
        print(f"Finished renaming {len(sorted_basenames)} files in {split} set.")
        
if __name__ == '__main__':
    # 请将这里的路径替换为你的数据集实际路径
    # 例如: 'E:/RoadDefect/dataset/Road crack detection and classification.v5i.yolov8'
    dataset_path = 'E:\dataset\Road crack detection and classification.v5i.yolov8'
    
    rename_dataset_files(dataset_path)
    print("\nAll files have been renamed.")