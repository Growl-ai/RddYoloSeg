import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import tempfile

# 设置页面配置
st.set_page_config(
    page_title="道路缺陷检测系统",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和描述
st.title("🛣️ YOLOv8道路缺陷检测系统")
st.markdown("""
使用YOLOv8分割模型检测道路缺陷，包括裂缝和坑洞等。
上传道路图像或视频，系统将自动识别并标注缺陷区域。
""")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    
    # 模型选择
    model_path = st.text_input("模型路径", value="E:\\RoadDefect\\models\\yolov8n-seg.pt")
    
    # 置信度阈值
    confidence = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.01)
    
    # 类别过滤
    st.subheader("检测类别")
    class_names = {
        0: "Crack-alligator",
        1: "Crack-long", 
        2: "Crack-trans",
        3: "pothole"
    }
    selected_classes = []
    for class_id, class_name in class_names.items():
        if st.checkbox(f"{class_name} ({class_id})", value=True):
            selected_classes.append(class_id)
    
    # 关于部分
    st.divider()
    st.subheader("ℹ️ 关于")
    st.markdown("""
    该应用使用YOLOv8分割模型检测道路缺陷。
    
    - **Crack-alligator**: 鳄鱼裂纹
    - **Crack-long**: 纵向裂缝
    - **Crack-trans**: 横向裂缝
    - **Pothole**: 坑洞
    """)

# 加载模型
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None

model = load_model(model_path)

# 文件上传区域
st.subheader("📤 上传图像或视频")
uploaded_file = st.file_uploader(
    "选择图像或视频文件", 
    type=['jpg', 'jpeg', 'png', 'bmp', 'mp4', 'avi', 'mov'],
    label_visibility="collapsed"
)

# 处理上传的文件
if uploaded_file is not None:
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
    
    # 判断文件类型
    if uploaded_file.type.startswith('image'):
        # 处理图像
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原图")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("检测结果")
            
            if model is not None:
                # 执行预测
                results = model.predict(
                    source=file_path,
                    conf=confidence,
                    classes=selected_classes,
                    save=False,
                    imgsz=640
                )
                
                # 绘制结果
                res_plotted = results[0].plot()
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_plotted_rgb, use_container_width=True)
                
                # 显示检测统计
                detections = results[0]
                if detections.boxes is not None:
                    num_detections = len(detections.boxes)
                    st.success(f"检测到 {num_detections} 个缺陷")
                    
                    # 显示每个检测的详细信息
                    for i, (box, mask, cls, conf) in enumerate(zip(
                        detections.boxes, 
                        detections.masks if detections.masks else [None]*len(detections.boxes),
                        detections.boxes.cls,
                        detections.boxes.conf
                    )):
                        class_name = class_names.get(int(cls), "未知")
                        st.write(f"{i+1}. {class_name}: {conf:.2f}")
                else:
                    st.warning("未检测到任何缺陷")
            else:
                st.error("模型未正确加载")
    
    elif uploaded_file.type.startswith('video'):
        # 处理视频
        st.subheader("视频检测结果")
        
        if model is not None:
            # 创建临时输出文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_tmp_file:
                output_path = output_tmp_file.name
            
            # 执行预测
            results = model.predict(
                source=file_path,
                conf=confidence,
                classes=selected_classes,
                save=True,
                project=tempfile.gettempdir(),
                name="yolov8_detection",
                exist_ok=True
            )
            
            # 查找输出视频文件
            output_video_path = os.path.join(tempfile.gettempdir(), "yolov8_detection", os.path.basename(file_path))
            
            if os.path.exists(output_video_path):
                # 显示视频
                st.video(output_video_path)
                
                # 提供下载链接
                with open(output_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                st.download_button(
                    label="下载处理后的视频",
                    data=video_bytes,
                    file_name="detected_video.mp4",
                    mime="video/mp4"
                )
            else:
                st.error("处理视频时出错")
        else:
            st.error("模型未正确加载")
    
    # 清理临时文件
    try:
        os.unlink(file_path)
    except:
        pass

else:
    # 显示示例图像
    st.info("👆 请上传图像或视频文件开始检测")
    
    # 示例图像
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://via.placeholder.com/300x200/4C7A8F/FFFFFF?text=道路图像示例1", 
                caption="示例图像 1", use_container_width=True)
    
    with col2:
        st.image("https://via.placeholder.com/300x200/4C7A8F/FFFFFF?text=道路图像示例2", 
                caption="示例图像 2", use_container_width=True)
    
    with col3:
        st.image("https://via.placeholder.com/300x200/4C7A8F/FFFFFF?text=道路图像示例3", 
                caption="示例图像 3", use_container_width=True)

# 页脚
st.divider()
st.markdown(
    "<div style='text-align: center; color: grey;'>道路缺陷检测系统 © 2025 | 基于YOLOv8分割模型</div>", 
    unsafe_allow_html=True
)