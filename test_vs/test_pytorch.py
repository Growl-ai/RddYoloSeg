# scripts/test_pytorch_enhanced.py
import torch
from ultralytics import YOLO
import time
import numpy as np
from pathlib import Path
import argparse
import json
import psutil
# import gc

def get_memory_usage():
    """获取内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    gpu_memory = 0
    gpu_utilization = 0
    try:
        # 尝试获取GPU内存和利用率信息
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory = info.used / 1024 / 1024
        
        # 获取GPU利用率
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = util.gpu
    except:
        pass
    
    return {
        'cpu_memory_mb': memory_info.rss / 1024 / 1024,
        'gpu_memory_mb': gpu_memory,
        'gpu_utilization': gpu_utilization
    }

def test_pytorch_model_enhanced(model_path, test_data, device='cuda', warmup=10, runs=100, batch_size=1):
    """
    增强版的PyTorch模型性能测试
    """
    # 记录初始内存使用
    initial_memory = get_memory_usage()
    
    # 加载模型
    model = YOLO(model_path)
    model.to(device)
    
    # 预热
    print(f"预热模型 ({warmup} 次推理)...")
    warmup_times = []
    for _ in range(warmup):
        start_time = time.time()
        results = model(test_data, verbose=False)
        end_time = time.time()
        warmup_times.append((end_time - start_time) * 1000)
    
    # 清空缓存
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 性能测试
    print(f"开始性能测试 ({runs} 次推理)...")
    times = []
    memory_usage = []
    gpu_utilizations = []
    
    for i in range(runs):
        # 记录内存使用和GPU利用率
        mem_util = get_memory_usage()
        memory_usage.append(mem_util)
        gpu_utilizations.append(mem_util['gpu_utilization'])
        
        start_time = time.time()
        results = model(test_data, verbose=False)
        end_time = time.time()
        
        times.append((end_time - start_time) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"已完成 {i+1}/{runs} 次推理")
    
    # 计算统计信息
    times = np.array(times)
    cpu_memory_usage = np.array([m['cpu_memory_mb'] for m in memory_usage])
    gpu_memory_usage = np.array([m['gpu_memory_mb'] for m in memory_usage])
    gpu_utilizations = np.array(gpu_utilizations)
    
    # 计算百分位数
    percentiles = {
        'p50': np.percentile(times, 50),
        'p90': np.percentile(times, 90),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
    
    return {
        'device': device,
        'batch_size': batch_size,  # 添加batch_size字段
        'avg_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'fps': 1000 / np.mean(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'total_time_ms': np.sum(times),
        'percentiles': percentiles,
        'memory_usage_mb': {
            'cpu_avg': np.mean(cpu_memory_usage),
            'cpu_max': np.max(cpu_memory_usage),
            'cpu_initial': initial_memory['cpu_memory_mb'],
            'gpu_avg': np.mean(gpu_memory_usage),
            'gpu_max': np.max(gpu_memory_usage),
            'gpu_initial': initial_memory['gpu_memory_mb']
        },
        'gpu_utilization_avg': np.mean(gpu_utilizations),
        'gpu_utilization_max': np.max(gpu_utilizations),
        'warmup_avg_ms': np.mean(warmup_times) if warmup_times else 0
    }

def convert_to_serializable(obj):
    """将NumPy数据类型转换为Python内置类型以确保JSON可序列化"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced PyTorch model performance test')
    parser.add_argument('--model', type=str, required=True, help='Path to the PyTorch model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data (image or directory)')
    parser.add_argument('--output', type=str, default='results/pytorch', help='Output directory for results')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=100, help='Number of test runs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试不同设备性能
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = {}
    for device in devices:
        print(f"测试{device.upper()}性能...")
        results[device] = test_pytorch_model_enhanced(
            args.model, args.data, device=device, 
            warmup=args.warmup, runs=args.runs, batch_size=args.batch_size
        )
    
    # 保存结果
    output_data = {
        'results': results,
        'model': args.model,
        'test_data': args.data,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'system_info': {
            'cpu': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
        }
    }
    
    # 生成结果文件名
    model_name = Path(args.model).stem
    output_file = output_dir / f"{model_name}_test5.json"
    
    # with open(output_file, 'w') as f:
    #     json.dump(output_data, f, indent=4)
    # 使用转换函数确保所有数据可序列化
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(output_data), f, indent=4)
    
    print(f"结果已保存到: {output_file}")
    # 打印简要结果
    print("\n性能测试结果:")
    for device, result in results.items():
        print(f"{device.upper()} - 批量大小 {result['batch_size']}:")
        print(f"  平均推理时间: {result['avg_time_ms']:.2f} ms")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"  内存使用: CPU {result['memory_usage_mb']['cpu_avg']:.1f} MB, "
              f"GPU {result['memory_usage_mb']['gpu_avg']:.1f} MB")
        if device == 'cuda':
            print(f"  GPU利用率: 平均 {result['gpu_utilization_avg']:.1f}%, "
                  f"最大 {result['gpu_utilization_max']:.1f}%")
        print()