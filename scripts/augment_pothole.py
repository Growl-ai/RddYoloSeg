# augment_pothole.py
import cv2
import numpy as np
import os
import random
import yaml
from pathlib import Path
from sklearn.utils import shuffle

class PotholeAugmentor:
    def __init__(self, dataset_path, output_path, target_count=1000):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_count = target_count
        
        # 创建输出目录
        (self.output_path / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'labels').mkdir(parents=True, exist_ok=True)
        
        self.load_dataset()
    
    def load_dataset(self):
        """加载数据集中的所有Pothole样本"""
        print("开始加载数据集...")
        self.pothole_samples = []  # 存储包含pothole的图像信息
        
        images_dir = self.dataset_path / 'train' / 'images'
        labels_dir = self.dataset_path / 'train' / 'labels'
        
        for img_path in images_dir.glob('*.jpg'):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
                
            # 读取图像和标注
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            pothole_bboxes = []
            other_bboxes = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    
                    if class_id == 3:  # Pothole类别
                        pothole_bboxes.append(bbox)
                    else:
                        other_bboxes.append({'class_id': class_id, 'bbox': bbox})
            
            if pothole_bboxes:
                self.pothole_samples.append({
                    'image': img,
                    'image_path': img_path,
                    'pothole_bboxes': pothole_bboxes,
                    'other_bboxes': other_bboxes,
                    'image_size': img.shape[:2]  # (height, width)
                })
        print(f"找到 {len(self.pothole_samples)} 个包含pothole的图像样本")
    def extract_pothole_patches(self):
        """从图像中提取Pothole patches"""
        print("开始提取pothole patches...")
        self.pothole_patches = []
        
        for sample in self.pothole_samples:
            img = sample['image']
            h, w = img.shape[:2]
            
            for bbox in sample['pothole_bboxes']:
                # 转换YOLO格式到像素坐标
                x_center, y_center, bbox_w, bbox_h = bbox
                x1 = int((x_center - bbox_w/2) * w)
                y1 = int((y_center - bbox_h/2) * h)
                x2 = int((x_center + bbox_w/2) * w)
                y2 = int((y_center + bbox_h/2) * h)
                
                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    patch = img[y1:y2, x1:x2].copy()
                    self.pothole_patches.append({
                        'patch': patch,
                        'original_bbox': bbox,
                        'source_size': (w, h)
                    })
        print(f"提取了 {len(self.pothole_patches)} 个pothole patches")
    def get_background_images(self):
        """获取不包含pothole的背景图像"""
        print("开始获取背景图像...")
        self.background_samples = []
        
        images_dir = self.dataset_path / 'train' / 'images'
        labels_dir = self.dataset_path / 'train' / 'labels'
        
        for img_path in images_dir.glob('*.jpg'):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            other_bboxes = []
            has_pothole = False
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                            
                            if class_id == 3:
                                has_pothole = True
                            else:
                                other_bboxes.append({'class_id': class_id, 'bbox': bbox})
            
            if not has_pothole:
                self.background_samples.append({
                    'image': img,
                    'other_bboxes': other_bboxes
                })
    
    def smart_paste(self, background_sample, pothole_patch):
        """智能粘贴并生成标签"""
        bg_img = background_sample['image']
        bg_h, bg_w = bg_img.shape[:2]
        patch = pothole_patch['patch']
        patch_h, patch_w = patch.shape[:2]
        
        # 随机选择粘贴位置
        margin = max(10, patch_w // 4, patch_h // 4)
        x = random.randint(margin, bg_w - patch_w - margin)
        y = random.randint(margin, bg_h - patch_h - margin)
        
        # 调整外观
        adjusted_patch = self.adjust_patch_appearance(patch, bg_img, x, y)
        
        # 混合到背景
        result = bg_img.copy()
        roi = result[y:y+patch_h, x:x+patch_w]
        
        # 使用alpha混合
        alpha = 0.7 + random.random() * 0.3  # 0.7-1.0之间的随机透明度
        result[y:y+patch_h, x:x+patch_w] = cv2.addWeighted(roi, 1-alpha, adjusted_patch, alpha, 0)
        
        # 生成YOLO格式标注
        x_center = (x + patch_w / 2) / bg_w
        y_center = (y + patch_h / 2) / bg_h
        width = patch_w / bg_w
        height = patch_h / bg_h
        
        # 收集所有标注（原有标注+新pothole标注）
        all_annotations = []
        
        # 添加原有标注
        for bbox_info in background_sample['other_bboxes']:
            cls_id = bbox_info['class_id']
            bbox = bbox_info['bbox']
            all_annotations.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        
        # 添加新pothole标注
        all_annotations.append(f"3 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return result, all_annotations
    
    def adjust_patch_appearance(self, patch, background, x, y):
        """调整patch外观以匹配背景"""
        patch_h, patch_w = patch.shape[:2]
        
        # 确保不越界
        if y + patch_h > background.shape[0] or x + patch_w > background.shape[1]:
            return patch
            
        bg_region = background[y:y+patch_h, x:x+patch_w]
        
        if bg_region.size == 0:
            return patch
        
        # 调整亮度和颜色匹配
        patch_mean = np.mean(patch)
        bg_mean = np.mean(bg_region)
        
        if patch_mean > 0:
            adjustment_factor = bg_mean / patch_mean
            adjusted = np.clip(patch * adjustment_factor, 0, 255).astype(np.uint8)
            return adjusted
        
        return patch
    
    def generate_augmented_samples(self):
        """生成增强样本"""
        print("加载数据集...")
        self.extract_pothole_patches()
        self.get_background_images()
        
        if not self.pothole_patches or not self.background_samples:
            print("错误：未找到足够的样本数据！")
            return
        
        print(f"找到 {len(self.pothole_patches)} 个pothole patches")
        print(f"找到 {len(self.background_samples)} 个背景图像")
        
        # 生成增强样本
        num_generated = 0
        
        while num_generated < self.target_count:
            # 随机选择背景和patch
            bg_sample = random.choice(self.background_samples)
            patch_info = random.choice(self.pothole_patches)
            
            try:
                augmented_img, annotations = self.smart_paste(bg_sample, patch_info)
                
                # 保存图像和标注
                output_img_path = self.output_path / 'images' / f'pothole_aug_{num_generated:04d}.jpg'
                output_label_path = self.output_path / 'labels' / f'pothole_aug_{num_generated:04d}.txt'
                
                cv2.imwrite(str(output_img_path), augmented_img)
                
                with open(output_label_path, 'w') as f:
                    for ann in annotations:
                        f.write(ann + '\n')
                
                num_generated += 1
                
                if num_generated % 50 == 0:
                    print(f"已生成 {num_generated} 个增强样本")
                    
            except Exception as e:
                print(f"生成样本时出错: {e}")
                continue
        
        print(f"完成！总共生成 {num_generated} 个pothole增强样本")
    
    def update_dataset_yaml(self):
        """更新数据集配置文件"""
        original_yaml_path = '../yolov8.yaml'
        
        if not original_yaml_path.exists():
            print("找不到原始data.yaml文件")
            return
        
        with open(original_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 更新训练路径
        original_train_path = data.get('train', '')
        if isinstance(original_train_path, list):
            new_train_paths = original_train_path.copy()
        else:
            new_train_paths = [original_train_path]
        
        # 添加增强数据路径
        new_train_paths.append(str(self.output_path / 'images'))
        
        data['train'] = new_train_paths
        
        # 保存新配置
        new_yaml_path = self.dataset_path / 'augmented_data.yaml'
        with open(new_yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"已创建增强数据集配置文件: {new_yaml_path}")

def main():
    dataset_path = "E:/RoadDefect/data/detect/processed"
    output_path = "E:/RoadDefect/data/detect/augmented_pothole"
    target_count = 500  # 目标生成数量
    
    augmentor = PotholeAugmentor(dataset_path, output_path, target_count)
    augmentor.generate_augmented_samples()
    augmentor.update_dataset_yaml()

if __name__ == "__main__":
    main()