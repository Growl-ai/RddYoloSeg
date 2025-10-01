# evaluate_model_unique.py
import os
os.environ['ULTRALYTICS_SETTINGS'] = 'no_git=True'
os.environ['ULTRAlytics_FONT'] = 'C:/Windows/Fonts/Arial.ttf'

from ultralytics import YOLO
from pathlib import Path
import argparse
from datetime import datetime

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

def evaluate_on_test_set(model_path, data_config, save_dir):
    """
    在测试集上评估模型性能
    
    Args:
        model_path: 模型路径
        data_config: 数据配置文件路径
        save_dir: 结果保存目录
    """
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = YOLO(str(model_path))
    
    # 在测试集上评估
    results = model.val(
        data=str(data_config),
        split='test',  # 指定使用测试集
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.6,
        save_json=True,
        save_conf=True,
        plots=True,
        project=str(save_dir.parent),  # 项目目录
        name=save_dir.name,            # 运行名称
        exist_ok=True                  # 允许覆盖现有结果
    )
    
    # 打印关键指标
    print("\n=== 测试集评估结果 ===")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"精确度: {results.box.mp:.4f}")
    print(f"召回率: {results.box.mr:.4f}")
    
    # # 保存详细结果到文本文件
    # result_file = save_dir / "evaluation_results.txt"
    # with open(result_file, 'w') as f:
    #     f.write("=== 测试集评估结果 ===\n")
    #     f.write(f"mAP50: {results.box.map50:.4f}\n")
    #     f.write(f"mAP50-95: {results.box.map:.4f}\n")
    #     f.write(f"精确度: {results.box.mp:.4f}\n")
    #     f.write(f"召回率: {results.box.mr:.4f}\n")
    #     f.write(f"\n各类别性能:\n")
    #     # 获取ap50列表
    #     ap50_list = results.box.ap50()  # 这是一个方法，返回列表
    #     # 确保ap50_list的长度和ap_class_index相同
    #     if len(ap50_list) != len(results.box.ap_class_index):
    #         print(f"警告: ap50列表长度({len(ap50_list)})与ap_class_index长度({len(results.box.ap_class_index)})不一致")
    #     # 使用正确的属性名称
    #     for i, class_idx in enumerate(results.box.ap_class_index):
    #         class_name = model.names[class_idx]
    #         # 检查索引i是否在ap50_list的范围内
    #         if i < len(ap50_list):
    #             ap50_value = ap50_list[i]
    #         else:
    #             ap50_value = 0.0  # 或者用其他方式处理
    #         f.write(f"{class_name}:\n")
    #         f.write(f"  mAP50: {ap50_value:.4f}\n")
    #         # f.write(f"  mAP50: {results.box.ap50()[i]:.4f}\n")  # 注意: ap50() 是一个方法
    #         f.write(f"  精确度: {results.box.p[i]:.4f}\n")      # 使用 p 而不是 class_mp
    #         f.write(f"  召回率: {results.box.r[i]:.4f}\n")      # 使用 r 而不是 class_mr
    #         f.write(f"  F1分数: {results.box.f1[i]:.4f}\n\n")   # 使用 f1
    
    # print(f"\n详细结果已保存至: {result_file}")
    return results, save_dir

def generate_unique_save_dir(base_dir, model_type, run_name):
    """
    生成唯一的保存目录路径
    
    Args:
        base_dir: 基础目录
        model_type: 模型类型
        run_name: 运行名称
        
    Returns:
        唯一的保存目录路径
    """
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建唯一目录名
    unique_dir_name = f"{run_name}_{timestamp}"
    
    # 返回完整路径
    return Path(base_dir) / unique_dir_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在测试集上评估YOLO模型')
    parser.add_argument('--model_type', type=str, default='detect', choices=['detect', 'seg'], 
                       help='模型类型: detect(检测)或seg(分割)')
    parser.add_argument('--model_size', type=str, default='s', 
                       help='模型大小: n, s, m等')
    parser.add_argument('--data_config', type=str, required=False, 
                       help='数据配置文件路径')
    parser.add_argument('--run_name', type=str, default='yolov8s_detect',
                       help='训练运行的名称')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录，如果指定则使用此目录，否则自动生成唯一目录')
    parser.add_argument('--base_save_dir', type=str, default=None,
                       help='基础保存目录，默认为 PROJECT_ROOT/infer/results')
    
    args = parser.parse_args()
    
    # 构建模型路径
    if args.model_type == 'detect':
        model_path = PROJECT_ROOT / 'runs' / 'detect' / args.run_name / 'weights' / 'best.pt'
        # data_config = PROJECT_ROOT / 'yolov8.yaml'
        data_config = 'E:\RoadDefect\data\detect\split\dataset.yaml'
    else:  # 分割模型
        model_path = PROJECT_ROOT / 'runs' / 'segment' / args.run_name / 'weights' / 'best.pt'
        # data_config = PROJECT_ROOT / 'yolov8_seg.yaml'
        data_config = 'E:\RoadDefect\data\seg\yolov8_seg.yaml'
    
    # 构建保存目录
    if args.save_dir is not None:
        # 使用用户指定的目录
        save_dir = Path(args.save_dir)
    else:
        # 使用基础保存目录
        if args.base_save_dir is None:
            base_save_dir = PROJECT_ROOT / 'infer' / 'results'
        else:
            base_save_dir = Path(args.base_save_dir)
        
        # 生成唯一目录
        save_dir = generate_unique_save_dir(base_save_dir, args.model_type, args.run_name)
    
    # 执行评估
    results, output_dir = evaluate_on_test_set(model_path, data_config, save_dir)
    
    print(f"\n评估完成! 所有结果保存在: {output_dir}")