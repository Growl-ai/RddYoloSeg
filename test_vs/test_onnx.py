# test_vs/test_onnx_improved.py
import onnxruntime as ort
import numpy as np
import cv2
import time
import json
from pathlib import Path
import argparse
import psutil
import glob
import os
import random
import traceback

# 自定义JSON编码器，用于处理NumPy数据类型------新增2
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)
        
def get_memory_usage():
    """获取内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    gpu_memory = 0
    try:
        # 尝试获取GPU内存信息（如果可用）
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = info.used / 1024 / 1024
    except ImportError:
        print("pynvml not installed, skipping GPU memory monitoring")
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
    
    return {
        'cpu_memory_mb': memory_info.rss / 1024 / 1024,
        'gpu_memory_mb': gpu_memory
    }

def get_gpu_utilization():
    """获取GPU利用率"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    except:
        return 0

def preprocess_image_yolov8(image_path, input_shape=(640, 640)):
    """
    YOLOv8专用的图像预处理
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 调整大小并保持宽高比
    h, w = image.shape[:2]
    scale = min(input_shape[0] / h, input_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    
    # 调整大小
    resized = cv2.resize(image, (nw, nh))
    
    # 填充到目标尺寸
    padded = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    
    # 转换为模型输入格式
    input_data = padded.astype(np.float32) / 255.0  # 归一化
    input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
    
    return input_data

def check_gpu_available():
    """检查GPU是否可用"""
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    
    if 'CUDAExecutionProvider' not in available_providers:
        print("CUDAExecutionProvider is not available")
        return False
    
    # 进一步检查CUDA设备
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {cuda_available}")
        if cuda_available:
            print(f"GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return cuda_available
    except ImportError:
        print("PyTorch not installed, skipping detailed GPU check")
        # 如果没有PyTorch，至少检查CUDAExecutionProvider是否可用
        return 'CUDAExecutionProvider' in available_providers

def test_onnx_model_comprehensive(model_path, test_images, providers=['CPUExecutionProvider'], 
                                 warmup=10, runs=100, input_shape=(640, 640), batch_sizes=[1, 4, 8]):
    """
    全面的ONNX模型性能测试，包括不同批量大小
    """
    # 检查GPU是否可用（如果请求的是CUDA provider）
    if 'CUDAExecutionProvider' in providers:
        if not check_gpu_available():
            raise RuntimeError("CUDAExecutionProvider requested but GPU is not available")
    
    # 记录初始内存使用
    initial_memory = get_memory_usage()
    
    # 创建ONNX运行时会话
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        print(f"Session created with providers: {session.get_providers()}")
    except Exception as e:
        print(f"创建ONNX会话失败: {e}")
        return None
    
    # 获取输入名称和形状
    input_name = session.get_inputs()[0].name
    input_shape_actual = session.get_inputs()[0].shape
    print(f"模型输入形状: {input_shape_actual}")
    
    # 检查模型支持的批量大小-----------新增1
    model_batch_size = input_shape_actual[0]
    if isinstance(model_batch_size, str) or model_batch_size == -1:
        # 动态批量大小
        print("模型支持动态批量大小")
    else:
        # 固定批量大小
        print(f"模型固定批量大小为: {model_batch_size}")
        # 过滤掉不支持的批量大小
        batch_sizes = [bs for bs in batch_sizes if bs == model_batch_size]
        if not batch_sizes:
            print(f"模型不支持任何指定的批量大小，使用默认批量大小1")
            batch_sizes = [1]
    
    print(f"将测试的批量大小: {batch_sizes}")

    # 预处理测试图像
    if isinstance(test_images, str):
        if os.path.isdir(test_images):
            image_files = glob.glob(os.path.join(test_images, "*.jpg")) + \
                         glob.glob(os.path.join(test_images, "*.png")) + \
                         glob.glob(os.path.join(test_images, "*.jpeg"))
            # 随机选择图片，确保多样性
            random.shuffle(image_files)
            test_images = image_files[:min(50, len(image_files))]  # 最多使用50张图片
        else:
            test_images = [test_images]
    
    print(f"使用 {len(test_images)} 张测试图片")
    
    # 预处理所有图片
    preprocessed_images = []
    for img_path in test_images:
        input_data = preprocess_image_yolov8(img_path, input_shape)
        preprocessed_images.append(input_data)
    
    results = {}
    
    # 测试不同批量大小
    for batch_size in batch_sizes:
        print(f"\n测试批量大小: {batch_size}")
        
        # 准备批量数据
        batched_data = []
        for i in range(0, len(preprocessed_images), batch_size):
            batch = preprocessed_images[i:i+batch_size]
            if len(batch) < batch_size:
                # 如果最后一批不足，用第一批填充
                batch.extend(preprocessed_images[:batch_size-len(batch)])
            batched_data.append(np.stack(batch))
        
        # 预热
        print(f"预热模型 ({warmup} 次推理)...")
        warmup_times = []
        for i in range(warmup):
            batch_idx = i % len(batched_data)
            input_data = batched_data[batch_idx]
            
            start_time = time.time()
            outputs = session.run(None, {input_name: input_data})
            end_time = time.time()
            warmup_time = (end_time - start_time) * 1000
            warmup_times.append(warmup_time)
            
            if (i + 1) % 5 == 0:
                print(f"预热进度: {i+1}/{warmup}, 最近推理时间: {warmup_time:.2f}ms")
        
        # 性能测试
        print(f"开始性能测试 ({runs} 次推理)...")
        times = []
        memory_usage = []
        gpu_utilization = []
        
        for i in range(runs):
            # 循环使用测试批次
            batch_idx = i % len(batched_data)
            input_data = batched_data[batch_idx]
            
            # 记录内存使用和GPU利用率
            if i % 10 == 0:  # 每10次记录一次，减少性能影响
                memory_usage.append(get_memory_usage())
                gpu_utilization.append(get_gpu_utilization())
            
            start_time = time.time()
            outputs = session.run(None, {input_name: input_data})
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000
            times.append(inference_time)
            
            if (i + 1) % 10 == 0:
                print(f"已完成 {i+1}/{runs} 次推理, 最近推理时间: {inference_time:.2f}ms")
        
        # 确保至少有一次内存记录
        if not memory_usage:
            memory_usage.append(get_memory_usage())
            gpu_utilization.append(get_gpu_utilization())
        
        # 计算统计信息
        times = np.array(times)
        cpu_memory_usage = np.array([m['cpu_memory_mb'] for m in memory_usage])
        gpu_memory_usage = np.array([m['gpu_memory_mb'] for m in memory_usage])
        
        # 计算百分位数
        percentiles = {
            'p50': np.percentile(times, 50),
            'p90': np.percentile(times, 90),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
        
        # results[batch_size] = {
        #     'avg_time_ms': np.mean(times),
        #     'std_time_ms': np.std(times),
        #     'fps': 1000 / np.mean(times) * batch_size,  # 考虑批量大小的FPS
        #     'min_time_ms': np.min(times),
        #     'max_time_ms': np.max(times),
        #     'total_time_ms': np.sum(times),
        #     'percentiles': percentiles,
        #     'memory_usage_mb': {
        #         'cpu_avg': np.mean(cpu_memory_usage),
        #         'cpu_max': np.max(cpu_memory_usage),
        #         'cpu_initial': initial_memory['cpu_memory_mb'],
        #         'gpu_avg': np.mean(gpu_memory_usage) if np.any(gpu_memory_usage > 0) else 0,
        #         'gpu_max': np.max(gpu_memory_usage) if np.any(gpu_memory_usage > 0) else 0,
        #         'gpu_initial': initial_memory['gpu_memory_mb']
        #     },
        #     'gpu_utilization_avg': np.mean(gpu_utilization) if gpu_utilization else 0,
        #     'gpu_utilization_max': np.max(gpu_utilization) if gpu_utilization else 0,
        #     'warmup_avg_ms': np.mean(warmup_times) if warmup_times else 0,
        #     'test_images_count': len(test_images),
        #     'batch_size': batch_size
        # }

        # 将NumPy类型转换为Python原生类型----------新增2
        results[batch_size] = {
            'avg_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'fps': float(1000 / np.mean(times) * batch_size),  # 考虑批量大小的FPS
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'total_time_ms': float(np.sum(times)),
            'percentiles': {k: float(v) for k, v in percentiles.items()},
            'memory_usage_mb': {
                'cpu_avg': float(np.mean(cpu_memory_usage)),
                'cpu_max': float(np.max(cpu_memory_usage)),
                'cpu_initial': float(initial_memory['cpu_memory_mb']),
                'gpu_avg': float(np.mean(gpu_memory_usage)) if np.any(gpu_memory_usage > 0) else 0,
                'gpu_max': float(np.max(gpu_memory_usage)) if np.any(gpu_memory_usage > 0) else 0,
                'gpu_initial': float(initial_memory['gpu_memory_mb'])
            },
            'gpu_utilization_avg': float(np.mean(gpu_utilization)) if gpu_utilization else 0,
            'gpu_utilization_max': float(np.max(gpu_utilization)) if gpu_utilization else 0,
            'warmup_avg_ms': float(np.mean(warmup_times)) if warmup_times else 0,
            'test_images_count': int(len(test_images)),
            'batch_size': int(batch_size)
        }
    
    return {
        'provider': providers[0],
        'batch_results': results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive ONNX model performance test')
    parser.add_argument('--model', type=str, required=True, help='Path to the ONNX model')
    parser.add_argument('--data', type=str, required=True, help='Path to test image or directory')
    parser.add_argument('--output', type=str, default='results/onnx', help='Output directory for results')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=100, help='Number of test runs')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[640, 640], 
                       help='Model input shape (height width)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8],
                       help='Batch sizes to test')
    parser.add_argument('--providers', type=str, nargs='+', default=['CPUExecutionProvider'], 
                       help='Execution providers to test')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试不同提供程序性能
    results = {}
    for provider in args.providers:
        print(f"\n{'='*50}")
        print(f"测试 {provider} 性能...")
        try:
            # 对于CUDA provider，强制检查GPU可用性
            if provider == 'CUDAExecutionProvider':
                if not check_gpu_available():
                    print("GPU not available, skipping CUDA test")
                    continue
                    
            result = test_onnx_model_comprehensive(
                args.model, args.data, providers=[provider], 
                warmup=args.warmup, runs=args.runs, 
                input_shape=tuple(args.input_shape),
                batch_sizes=args.batch_sizes
            )
            if result:
                results[provider] = result
                # 打印每个批量大小的结果
                for batch_size, batch_result in result['batch_results'].items():
                    print(f"{provider} (batch={batch_size}): "
                          f"平均推理时间 {batch_result['avg_time_ms']:.2f}ms, "
                          f"FPS: {batch_result['fps']:.2f}")
        except Exception as e:
            print(f"{provider} 测试失败: {e}")
    
    # 保存结果
    output_data = {
        'results': results,
        'model': args.model,
        'test_data': args.data,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'system_info': {
            'cpu': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
        },
        'test_config': {
            'warmup': args.warmup,
            'runs': args.runs,
            'input_shape': args.input_shape,
            'batch_sizes': args.batch_sizes
        }
    }
    
    # 生成结果文件名
    model_name = Path(args.model).stem
    output_file = output_dir / f"{model_name}_test_improved.json"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 打印简要结果
    print("\n性能测试结果:")
    print("=" * 80)
    for provider, result in results.items():
        print(f"{provider}:")
        for batch_size, batch_result in result['batch_results'].items():
            print(f"  批量大小 {batch_size}:")
            print(f"    平均推理时间: {batch_result['avg_time_ms']:.2f} ms")
            print(f"    FPS: {batch_result['fps']:.2f}")
            print(f"    内存使用: CPU {batch_result['memory_usage_mb']['cpu_avg']:.1f} MB, "
                  f"GPU {batch_result['memory_usage_mb']['gpu_avg']:.1f} MB")
            print(f"    GPU利用率: 平均 {batch_result['gpu_utilization_avg']:.1f}%, "
                  f"最大 {batch_result['gpu_utilization_max']:.1f}%")
            print()