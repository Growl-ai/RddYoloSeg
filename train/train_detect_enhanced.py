# train/train_detect_enhanced.py
import os
os.environ['ULTRALYTICS_SETTINGS'] = 'no_git=True'
os.environ['ULTRAlytics_FONT'] = 'C:/Windows/Fonts/Arial.ttf'

from ultralytics import YOLO
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent

def train_yolov8_detect_enhanced(model_size):
    """增强版的训练函数，包含copy-paste增强"""
    
    # 加载最佳模型
    model = YOLO(str(PROJECT_ROOT / 'runs' / 'detect' / 'yolov8m_simple' / 'weights' / 'best.pt'))
    
    # 使用增强后的数据配置
    data_yaml_path = PROJECT_ROOT / 'data' / 'augmented_data.yaml'
    
    # 训练参数
    results = model.train(
        data=str(data_yaml_path),
        epochs=150,
        imgsz=640,
        batch=-1,
        patience=15,
        save_period=20,
        name=f'yolov8n_detect_enhanced',
        device=0,
        amp=False,
        # 优化参数
        lr0=0.0001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        cos_lr=True,
        label_smoothing=0.1,
        # 数据增强
        copy_paste=0.3,  # 30%概率使用copy-paste
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.001,
        fliplr=0.5,
        # 类别权重
        #cls_pw=[1.0, 1.0, 1.2, 2.0]
    )
    
    return results, 'yolov8n_detect_enhanced'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train enhanced YOLOv8 detection model')
    parser.add_argument('--model_size', type=str, default='n', help='Model size (n, s, or m)')
    args = parser.parse_args()
    
    results, model_name = train_yolov8_detect_enhanced(args.model_size)
    print(f"Enhanced training completed. Model saved at: {PROJECT_ROOT / 'runs' / 'detect' / model_name}")