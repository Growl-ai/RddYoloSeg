# RDDYOLOSEG - Road Defect Detection & Segmentation

端到端的道路缺陷智能检测与分割系统，基于YOLOv8实现裂缝、坑洼等道路缺陷的自动识别与精确定位。

## 🚀 项目特色

- **创新半自动标注流程**: 采用"检测辅助标注，分割迭代优化"的两阶段数据生产流程，显著降低标注成本
- **轻量高效模型**: 基于YOLOv8n-seg的轻量级实例分割模型，平衡精度与速度
- **完整工程化**: 从数据准备到Web部署的全流程实现
- **跨平台支持**: 支持PyTorch/ONNX格式，Python/C++推理

## 📁 项目结构

```
RDDYOLOSEG/
├── data/                    # 数据目录
│   ├── detect/             # 检测任务数据（基于RDD2022）
│   │   ├── train/         # 训练集
│   │   ├── val/           # 验证集  
│   │   └── test/          # 测试集
│   └── seg/               # 分割任务数据（基于自有数据集）
│       ├── train/         # 训练集
│       ├── val/           # 验证集
│       └── test/          # 测试集
├── scripts/                # 数据处理脚本
│   ├── xml_to_yolo.py            # XML转YOLO格式（RDD2022）
│   ├── detect_data_analysis.py   # 检测数据分析
│   ├── seg_data_analysis.py      # 分割数据分析
│   ├── split_detect_data.py      # 检测数据划分
│   ├── split_seg_data.py         # 分割数据划分
│   ├── rename_seg_data.py        # 数据重命名
│   └── augment_pothole.py        # 数据增强
├── train/                  # 训练相关
│   ├── train_detect.py           # 检测模型训练（RDD2022）
│   ├── train_detect_enhanced.py  # 增强检测训练
│   ├── train_seg.py              # 分割模型训练（自有数据集）
│   └── train_seg113.py           # 特定分割训练
├── infer/                  # 推理相关
│   └── evaluate.py         # 模型评估
├── models/                 # 模型文件
│   ├── yolov8n-seg.pt     # PyTorch模型
│   └── yolov8n-seg.onnx   # ONNX模型
├── test_vs/               # 性能测试
│   ├── test_pytorch.py    # PyTorch推理测试
│   ├── test_onnx.py       # ONNX推理测试
│   ├── export_models.py   # 模型导出
│   └── val_convert.py     # 验证转换
│   └── TestOnnx.cpp       # ONNX推理测试
├── web/                   # Web应用
│   └── app_streamlit.py   # Streamlit应用
├── test_images/           # 测试图片
├── requirements.txt       # 依赖列表
└── README.md             # 项目说明
```

## ⚙️ 环境配置

### PyTorch安装
```bash
# 安装指定版本的PyTorch
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 其他依赖
```bash
pip install -r requirements.txt
```

## 🔧 环境依赖说明

### 训练阶段依赖
- **PyTorch GPU版本**：已包含必要的CUDA运行时，无需单独安装CUDA工具包
- **验证命令**：
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

### 推理阶段依赖
- **CUDA工具包**：ONNX Runtime需要系统级CUDA支持
- **cuDNN**：深度学习加速库
- **验证命令**：
  ```bash
  python -c "import onnxruntime as ort; print('CUDA' in ort.get_available_providers())"
  ```

### 版本兼容性
本项目测试环境：
- **CUDA Toolkit**: 12.4
- **cuDNN**: 9.4
- **PyTorch**: 2.5.1+cu121
- **ONNX Runtime-gpu**: 1.19.2

## 📊 数据集与标注流程

### 数据源说明
- **RDD2022公开数据集**: 包含XML格式标注，用于训练初始检测模型
- **自有道路数据集**: 无标注原始图像，用于最终分割模型训练

### 半自动标注流程
1. **检测模型预训练**:
   - 使用RDD2022数据集（XML标签转换为YOLO格式）训练YOLOv8检测模型
   - 脚本：`scripts/xml_to_yolo.py`

2. **边界框预标注**:
   - 使用训练好的检测模型对自有数据集进行边界框预标注
   - 人工校验修正，迭代优化检测模型至mAP@0.94

3. **分割掩码标注**:
   - 在优质边界框基础上，使用LabelMe对图像进行分割掩码精标
   - JSON格式转换为YOLO分割格式（使用labelme2yolo工具）

## 🛠️ 使用方法

### 1. 数据准备与格式转换
```bash
# RDD2022数据格式转换（XML → YOLO）
python scripts/xml_to_yolo.py

# 数据分析与划分
python scripts/detect_data_analysis.py
python scripts/split_detect_data.py
```

### 2. 检测模型训练（RDD2022）
```bash
python train/train_detect.py
```

### 3. 自有数据标注流程
```bash
# 使用训练好的检测模型进行预标注
# 人工校验后，进行分割数据准备
python scripts/split_seg_data.py
python scripts/seg_data_analysis.py
```

### 4. 分割模型训练（自有数据集）
```bash
python train/train_seg.py
```

### 5. 模型评估与优化
```bash
python infer/evaluate.py
```

### 6. 模型导出与部署
```bash
# 导出ONNX模型
python test_vs/export_models.py

# 性能对比
python test_vs/test_pytorch.py
python test_vs/test_onnx.py

# Web应用
cd web
streamlit run app_streamlit.py
```

## 📈 模型性能

| 模型 | 任务 | 数据源 | mAP50 | 用途 |
|------|------|--------|-------|------|
| YOLOv8m | 检测 | RDD2022 | 0.94 | 预标注自有数据 |
| YOLOv8n-seg | 分割 | 自有数据 | 0.91 | 最终部署模型 |

## 📊 性能基准测试

### 推理性能对比

我们对不同环境下的模型推理性能进行了全面测试，结果如下：

| 测试环境 | 推理后端 | 设备 | 平均推理时间 (ms) | FPS | 峰值内存 | 关键观察与优化效果 |
|---------|---------|------|------------------|-----|----------|-------------------|
| **Python** | PyTorch | CPU | 5873.74 | 0.17 | ~1.5 GB | ⚠️ **性能瓶颈**，无法满足实时需求 |
| **Python** | PyTorch | CUDA | 1107.46 | 0.90 | ~2 GB CPU / ~2 GB GPU | 🔍 Python解释器开销显著 |
| **Python** | ONNX | CPU | 29.57 | 33.82 | ~165 MB | ✅ **显著优化**，相比PyTorch CPU提升**200倍** |
| **Python** | ONNX | CUDA | **7.48** | **133.73** | ~815 MB CPU / ~1 GB GPU | 🏆 **Python最佳方案**，达到实时处理标准 |
| **C++** | ONNX | CPU | 27.44 | 36.44 | ~224 MB | 🔧 内存管理更优，性能稳定 |
| **C++** | ONNX | CUDA | **5.52** | **181.29** | ~1.4 GB GPU | 🚀 **全局最优**，延迟最低，吞吐量最高 |

### 关键结论

1. **ONNX Runtime相比原生PyTorch有显著性能优势**
   - Python环境下提升约**200倍** (CPU)
   - 内存使用减少**90%** (CPU场景)

2. **C++实现相比Python有进一步优化**
   - 延迟降低约**26%** (CUDA场景)
   - 内存管理更加高效

3. **推荐部署方案**
   - **高性能需求**: C++ + ONNX + CUDA (181 FPS)
   - **快速开发**: Python + ONNX + CUDA (134 FPS)  
   - **边缘设备**: Python/C++ + ONNX + CPU (34-36 FPS)

### 测试配置
- **硬件**: NVIDIA RTX 2080 GPU, Intel i7-12700K CPU
- **软件**: ONNX Runtime 1.19.2, PyTorch 2.5.1, CUDA 12.1
- **模型**: YOLOv8n-seg (输入尺寸: 640×640)
- **测试数据**: 100次推理求平均

## web演示
![web-s3](https://github.com/user-attachments/assets/0f203b80-1be6-4687-8a73-08717751d4d1)

![web-s4](https://github.com/user-attachments/assets/4e786412-290d-484e-a7ab-18d74ad413ad)

<img width="1272" height="1027" alt="web" src="https://github.com/user-attachments/assets/25d5b05f-5f93-47d6-a3f3-2c125d82e3b9" />

## 🔧 技术栈

- **深度学习框架**: PyTorch 2.5.1
- **模型架构**: YOLOv8
- **标注工具**: LabelMe
- **格式转换**: labelme2yolo, 自定义XML转YOLO脚本
- **推理引擎**: ONNX Runtime
- **Web框架**: Streamlit

## 📝 许可证

本项目仅供学习交流使用。

## 🤝 贡献

欢迎提交Issue和Pull Request！
