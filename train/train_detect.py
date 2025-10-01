# train_detect_simple.py
import os
os.environ['ULTRALYTICS_SETTINGS'] = 'no_git=True'
os.environ['ULTRAlytics_FONT'] = 'C:/Windows/Fonts/Arial.ttf'

from ultralytics import YOLO
from pathlib import Path
import argparse

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

def train_yolov8_detection_simple(model_path, data_yaml_path, epochs=150, imgsz=640, batch=-1, patience=20, resume=False):
    """
    简洁版YOLOv8检测模型训练
    
    Args:
        model_path: 预训练模型路径
        data_yaml_path: 数据配置文件路径
        epochs: 训练轮数
        imgsz: 图像尺寸
        batch: 批次大小
        patience: 早停耐心值
        resume: 是否从上次训练中断处继续训练
    """
    # 验证模型文件是否存在
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return None, None
    
    # 验证数据配置文件是否存在
    if not Path(data_yaml_path).exists():
        print(f"错误: 数据配置文件不存在: {data_yaml_path}")
        return None, None
    
    # 加载预训练模型
    model = YOLO(model_path)
    
    # 生成运行名称
    model_name = Path(model_path).stem
    run_name = f"{model_name}_simple"
    
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
            amp=False,
            resume=resume
        )
        
        print(f"\n训练完成! 最佳模型保存在: {PROJECT_ROOT / 'runs' / 'detect' / run_name}")
        return results, run_name
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='简洁版YOLOv8检测模型训练')
    parser.add_argument('--model_path', type=str, default=str(PROJECT_ROOT / 'models' / 'detect' / 'yolov8m.pt'),
                       help='预训练模型路径')
    parser.add_argument('--data_yaml', type=str, default=str(PROJECT_ROOT / 'data' / 'detect' / 'split' / 'dataset.yaml'),
                       help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=-1, help='批次大小')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--resume', action='store_true', help='从上次训练中断处继续训练')
    
    args = parser.parse_args()
    
    # 训练模型
    results, run_name = train_yolov8_detection_simple(
        model_path=args.model_path,
        data_yaml_path=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        resume=args.resume
    )
    
    if run_name:
        print(f"\n训练完成! 运行名称: {run_name}")
    else:
        print("\n训练失败!")