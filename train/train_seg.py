# train_seg.py
import os
os.environ['ULTRALYTICS_SETTINGS'] = 'no_git=True'
os.environ['ULTRAlytics_FONT'] = 'C:/Windows/Fonts/Arial.ttf'

from ultralytics import YOLO
from pathlib import Path
import argparse

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

def train_yolov8_segmentation_simple(model_size='n', epochs=150, imgsz=640, batch=-1, patience=20):
    """
    简洁版YOLOv8分割模型训练
    
    Args:
        model_size: 模型大小 'n', 's', 'm', 'l', 'x'
        epochs: 训练轮数
        imgsz: 图像尺寸
        batch: 批次大小
        patience: 早停耐心值
    """
    # 模型映射字典
    model_map = {
        'n': 'yolov8n-seg.pt',
        's': 'yolov8s-seg.pt', 
        'm': 'yolov8m-seg.pt',
        'l': 'yolov8l-seg.pt',
        'x': 'yolov8x-seg.pt'
    }
    
    if model_size not in model_map:
        raise ValueError("不支持的模型大小。请选择 'n', 's', 'm', 'l' 或 'x'")
    
    # 构建数据配置文件路径
    data_yaml_path = PROJECT_ROOT / 'data' / 'seg' / 'yolov8_seg.yaml'
    
    # 验证数据配置文件是否存在
    if not data_yaml_path.exists():
        print(f"错误: 数据配置文件不存在: {data_yaml_path}")
        return None, None
    
    # 加载预训练模型
    model_path = PROJECT_ROOT / 'models' / 'seg' / model_map[model_size]

    model = YOLO(str(model_path))
    
    # 生成运行名称
    run_name = f"yolov8{model_size}-seg_simple"
    
    # 训练模型 - 简洁版配置
    print(f"\n开始简洁版训练 {run_name}...")
    
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            save_period=20,
            name=run_name,
            device=0,  # 使用GPU
            amp=False
        )
        
        print(f"\n简洁版训练完成! 最佳模型保存在: {PROJECT_ROOT / 'runs' / 'segment' / run_name}")
        return results, run_name
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='简洁版YOLOv8分割模型训练')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小: n, s, m, l, x')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=-1, help='批次大小')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    
    args = parser.parse_args()
    
    # 训练模型
    results, run_name = train_yolov8_segmentation_simple(
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience
    )
    
    print(f"\n简洁版训练完成! 运行名称: {run_name}")