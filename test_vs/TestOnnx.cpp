﻿// TestOnnx.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <psapi.h>

// 获取当前进程内存使用（MB）
size_t get_current_memory_usage() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.PrivateUsage / (1024 * 1024);
}

// YOLOv8专用的图像预处理
cv::Mat preprocess_image_yolov8(const cv::Mat& image, const cv::Size& target_size) {
    int h = image.rows;
    int w = image.cols;
    float scale = min(target_size.height / (float)h, target_size.width / (float)w);
    int nh = static_cast<int>(h * scale);
    int nw = static_cast<int>(w * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh));

    cv::Mat padded = cv::Mat::zeros(target_size, CV_8UC3);
    padded.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(0, 0, nw, nh)));

    cv::Mat float_img;
    padded.convertTo(float_img, CV_32F, 1.0 / 255.0);

    return float_img;
}

// 测试ONNX模型
void test_onnx_model(const std::string& model_path, const std::vector<std::string>& image_paths,
    const std::string& provider, int warmup, int runs) {
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "test");
    Ort::SessionOptions session_options;

    if (provider == "CUDA") {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    }

    size_t initial_memory = get_current_memory_usage();

    std::wstring wmodel_path(model_path.begin(), model_path.end());
    Ort::Session session(env, wmodel_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_names;
    std::vector<int64_t> input_shape;
    
    size_t num_input_nodes = session.GetInputCount();
    std::cout << "Number of inputs: " << num_input_nodes << std::endl;
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name_ptr = session.GetInputNameAllocated(i, allocator);
        input_names_ptr.push_back(std::move(input_name_ptr));
        input_names.push_back(input_names_ptr.back().get());
        std::cout << "Input " << i << " name: " << input_names.back() << std::endl;
        
        if (i == 0) {
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape = tensor_info.GetShape();
        }
    }

    std::cout << "模型输入形状: [";
    for (size_t i = 0; i < input_shape.size(); i++) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::vector<std::vector<float>> processed_images;
    for (const auto& image_path : image_paths) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "无法读取图像: " << image_path << std::endl;
            continue;
        }

        cv::Mat processed = preprocess_image_yolov8(image,
            cv::Size(static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2])));

        cv::Mat channels[3];
        cv::split(processed, channels);
        std::vector<float> input_data;
        for (int c = 0; c < 3; c++) {
            input_data.insert(input_data.end(),
                (float*)channels[c].data,
                (float*)channels[c].data + input_shape[2] * input_shape[3]);
        }
        processed_images.push_back(input_data);
    }

    if (processed_images.empty()) {
        std::cerr << "没有有效的测试图像" << std::endl;
        return;
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    std::vector<const char*> output_names;
    size_t num_output_nodes = session.GetOutputCount();
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_names_ptr.push_back(std::move(output_name));
        output_names.push_back(output_names_ptr.back().get());
        std::cout << "Output " << i << " name: " << output_names.back() << std::endl;
    }

    if (output_names.empty()) {
        std::cerr << "错误: 没有找到输出名称" << std::endl;
        return;
    }

    for (size_t i = 0; i < output_names.size(); i++) {
        if (output_names[i] == nullptr || strlen(output_names[i]) == 0) {
            std::cerr << "错误: 输出名称 " << i << " 为空" << std::endl;
            return;
        }
    }

    std::cout << "预热模型 (" << warmup << " 次推理)..." << std::endl;
    std::vector<double> warmup_times;
    for (int i = 0; i < warmup; i++) {
        const auto& input_data = processed_images[i % processed_images.size()];
        input_tensors.clear();
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input_data.data()),
            input_data.size(), input_shape.data(), input_shape.size()));

        auto start = std::chrono::high_resolution_clock::now();
        try {
            session.Run(Ort::RunOptions{ nullptr },
                input_names.data(), input_tensors.data(), input_names.size(),
                output_names.data(), output_names.size());
        }
        catch (const std::exception& e) {
            std::cerr << "Exception during model run: " << e.what() << std::endl;
            return;
        }
        catch (...) {
            std::cerr << "Unknown exception during model run." << std::endl;
            return;
        }
        auto end = std::chrono::high_resolution_clock::now();

        warmup_times.push_back(std::chrono::duration<double>(end - start).count() * 1000);
    }

    std::cout << "开始性能测试 (" << runs << " 次推理)..." << std::endl;
    std::vector<double> times;
    std::vector<size_t> memory_usage;

    for (int i = 0; i < runs; i++) {
        memory_usage.push_back(get_current_memory_usage());

        const auto& input_data = processed_images[i % processed_images.size()];
        input_tensors.clear();
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input_data.data()),
            input_data.size(), input_shape.data(), input_shape.size()));

        auto start = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{ nullptr },
            input_names.data(), input_tensors.data(), input_names.size(),
            output_names.data(), output_names.size());
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration<double>(end - start).count() * 1000);

        if ((i + 1) % 10 == 0) {
            std::cout << "已完成 " << (i + 1) << "/" << runs << " 次推理" << std::endl;
        }
    }

    double total_time = 0;
    double min_time = (std::numeric_limits<double>::max)();
    double max_time = 0;
    for (auto t : times) {
        total_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    double avg_time = total_time / runs;
    double fps = 1000 / avg_time;

    size_t total_memory = 0;
    size_t max_memory = 0;
    for (auto m : memory_usage) {
        total_memory += m;
        if (m > max_memory) max_memory = m;
    }
    double avg_memory = static_cast<double>(total_memory) / memory_usage.size();

    std::cout << "ONNX Runtime (" << provider << ") 性能测试结果:" << std::endl;
    std::cout << "平均推理时间: " << avg_time << " ms" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "最小推理时间: " << min_time << " ms" << std::endl;
    std::cout << "最大推理时间: " << max_time << " ms" << std::endl;
    std::cout << "平均内存使用: " << avg_memory << " MB" << std::endl;
    std::cout << "最大内存使用: " << max_memory << " MB" << std::endl;
    std::cout << "初始内存使用: " << initial_memory << " MB" << std::endl;
}

int main() {
    try {
        std::string onnx_model_path = "models/yolov8n-seg.onnx";

        std::vector<std::string> image_paths;
        image_paths.push_back("E:/TestYolo/TestYolo/images/test_image.jpg");

        int warmup = 10;
        int runs = 100;

        std::cout << "测试ONNX模型在CPU上..." << std::endl;
        test_onnx_model(onnx_model_path, image_paths, "CPU", warmup, runs);

        std::cout << "测试ONNX模型在GPU上..." << std::endl;
        test_onnx_model(onnx_model_path, image_paths, "CUDA", warmup, runs);
    }
    catch (const std::exception& e) {
        std::cerr << "程序异常: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "未知异常" << std::endl;
        return 1;
    }
    return 0;
}