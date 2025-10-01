import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def test_yolov8_seg_model(specific_image_path=None):
    # 设置路径
    pt_model_path = "E:/RoadDefect/models/yolov8n-seg.pt"
    onnx_model_path = "E:/RoadDefect/runs/segment/yolov8n-seg_simple/weights/best.onnx"
    test_image_dir = "E:/RoadDefect/test_images"
    
    # 加载测试图像
    if specific_image_path:
        # 使用指定的图像路径
        image_path = Path(specific_image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"指定的图像不存在: {specific_image_path}")
        print(f"使用指定测试图像: {image_path}")
    else:
        # 使用测试目录中的图像
        test_images = list(Path(test_image_dir).glob("*.jpg")) + list(Path(test_image_dir).glob("*.png"))
        if not test_images:
            raise ValueError(f"在 {test_image_dir} 中未找到图像")
        
        # 选择第一张测试图像
        image_path = test_images[0]
        print(f"使用测试图像: {image_path}")
    
    # 加载并预处理图像
    original_image = Image.open(image_path).convert('RGB')
    
    # YOLOv8 的预处理方式
    def preprocess(image):
        # 调整大小到 YOLOv8 的输入尺寸 (640x640)
        image = image.resize((640, 640))
        # 转换为 numpy 数组并归一化
        image = np.array(image).astype(np.float32) / 255.0
        # 转换维度顺序为 CHW
        image = image.transpose(2, 0, 1)
        # 添加批次维度
        image = np.expand_dims(image, axis=0)
        return image
    
    input_tensor = preprocess(original_image)
    
    # 测试 PyTorch 模型
    print("测试 PyTorch 模型...")
    
    # 使用 Ultralytics YOLO 类加载模型
    try:
        from ultralytics import YOLO
        pt_model = YOLO(pt_model_path)
        pt_model = pt_model.model  # 获取实际的模型对象
        pt_model.eval()
    except ImportError:
        raise ImportError("需要安装 ultralytics 库: pip install ultralytics")
    
    with torch.no_grad():
        pt_output = pt_model(torch.from_numpy(input_tensor))
    
    # 测试 ONNX 模型
    print("测试 ONNX 模型...")
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # 打印ONNX模型输入输出信息
    print("ONNX模型输入:")
    for i, input in enumerate(ort_session.get_inputs()):
        print(f"  输入 {i}: {input.name}, 形状: {input.shape}")
    
    print("ONNX模型输出:")
    for i, output in enumerate(ort_session.get_outputs()):
        print(f"  输出 {i}: {output.name}, 形状: {output.shape}")
    
    # ONNX 运行时输入名称
    input_name = ort_session.get_inputs()[0].name
    
    # 运行推理
    onnx_output = ort_session.run(None, {input_name: input_tensor})
    
    # 比较输出
    print("\n比较输出...")
    
    # 处理 PyTorch 输出，将其转换为扁平列表
    def flatten_outputs(outputs):
        flat_list = []
        for output in outputs:
            if isinstance(output, (list, tuple)):
                flat_list.extend(flatten_outputs(output))
            elif isinstance(output, torch.Tensor):
                flat_list.append(output)
        return flat_list
    
    pt_output_flat = flatten_outputs(pt_output)
    print(f"PyTorch 输出扁平化后有 {len(pt_output_flat)} 个张量")
    for i, output in enumerate(pt_output_flat):
        print(f"  PyTorch 输出 {i}: 形状 {output.shape}")
    
    print(f"ONNX 输出有 {len(onnx_output)} 个张量")
    for i, output in enumerate(onnx_output):
        print(f"  ONNX 输出 {i}: 形状 {output.shape}")
    
    # 比较对应位置的输出
    min_outputs = min(len(pt_output_flat), len(onnx_output))
    for i in range(min_outputs):
        print(f"\n比较输出 {i}:")
        
        pt_out = pt_output_flat[i]
        onnx_out = onnx_output[i]
        
        if isinstance(pt_out, torch.Tensor):
            pt_out_np = pt_out.numpy()
            print(f"  PyTorch 形状: {pt_out_np.shape}")
            print(f"  ONNX 形状: {onnx_out.shape}")
            
            # 检查形状是否匹配
            if pt_out_np.shape == onnx_out.shape:
                # 计算差异
                diff = np.abs(pt_out_np - onnx_out)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"  最大差异: {max_diff}")
                print(f"  平均差异: {mean_diff}")
                
                # 设置可接受的误差阈值
                if max_diff < 1e-2 and mean_diff < 1e-3:
                    print("  ✓ 输出在可接受容差范围内匹配")
                else:
                    print("  ✗ 输出差异显著")
            else:
                print("  ✗ 输出形状不匹配")
        else:
            print(f"  PyTorch 输出类型: {type(pt_out)}")
    
    # 如果有不匹配的输出数量
    if len(pt_output_flat) != len(onnx_output):
        print(f"\n警告: PyTorch 和 ONNX 输出数量不匹配 ({len(pt_output_flat)} vs {len(onnx_output)})")
        if len(pt_output_flat) > len(onnx_output):
            print("  多余的 PyTorch 输出:")
            for i in range(len(onnx_output), len(pt_output_flat)):
                if isinstance(pt_output_flat[i], torch.Tensor):
                    print(f"    输出 {i}: 形状 {pt_output_flat[i].shape}")
                else:
                    print(f"    输出 {i}: 类型 {type(pt_output_flat[i])}")
        else:
            print("  多余的 ONNX 输出:")
            for i in range(len(pt_output_flat), len(onnx_output)):
                print(f"    输出 {i}: 形状 {onnx_output[i].shape}")
    
    # 可视化结果（可选）
    visualize_results(original_image, pt_output_flat, onnx_output, input_tensor.shape[2:])
    
    print("\n测试完成!")

def visualize_results(original_image, pt_output, onnx_output, input_shape):
    """可视化 PyTorch 和 ONNX 模型的预测结果"""
    # 将图像调整回原始大小
    resized_image = original_image.resize(input_shape)
    
    # 创建可视化图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    axes[0].imshow(resized_image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示 PyTorch 模型输出
    axes[1].imshow(resized_image)
    axes[1].set_title('PyTorch 预测')
    axes[1].axis('off')
    
    # 显示 ONNX 模型输出
    axes[2].imshow(resized_image)
    axes[2].set_title('ONNX 预测')
    axes[2].axis('off')
    
    # 添加简单的文本说明
    if len(pt_output) > 0 and isinstance(pt_output[0], np.ndarray):
        fig.text(0.15, 0.02, f"PyTorch 输出形状: {pt_output[0].shape}", ha='center')
    if len(onnx_output) > 0:
        fig.text(0.85, 0.02, f"ONNX 输出形状: {onnx_output[0].shape}", ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

if __name__ == "__main__":
    # 方法1: 使用测试目录中的第一张图像
    # test_yolov8_seg_model()
    
    # 方法2: 指定特定图像路径
    specific_image_path = "E:/RoadDefect/test_images/test_00041.jpg"  # 替换为你的图像路径
    test_yolov8_seg_model(specific_image_path)