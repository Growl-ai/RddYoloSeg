# test_vs/export_models.py
import os
os.environ['ULTRALYTICS_AUTOINSTALL'] = 'False'
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse

def export_models(model_path, output_dir):
    """
    导出模型为ONNX和TorchScript格式
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取模型名称
    model_name = Path(model_path).stem
    
    # 导出ONNX格式
    model_onnx = YOLO(model_path)
    onnx_path = output_dir / f"{model_name}.onnx"
    model_onnx.export(format="onnx", imgsz=640, simplify=True, opset=12, name=model_name, project=output_dir)
    
    # 导出TorchScript格式
    model_ts = YOLO(model_path)
    torchscript_path = output_dir / f"{model_name}.torchscript"
    model_ts.export(format="torchscript", imgsz=640, name=model_name, project=output_dir)
    
    print(f"模型已导出:")
    print(f"  ONNX: {onnx_path}")
    print(f"  TorchScript: {torchscript_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export trained models to ONNX and TorchScript formats')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output', type=str, default='exported_models', help='Output directory for exported models')
    
    args = parser.parse_args()
    
    export_models(args.model, args.output)