# scripts/xml_to_yolo.py

import os
import xml.etree.ElementTree as ET

# 定义类别名称到整数ID的映射
# 根据你的数据集标签 {D00, D10, D20, D40} 确定
classes = {
    'D00': 0,
    'D10': 1,
    'D20': 2,
    'D40': 3,
}

def convert_xml_to_yolo(xml_file_path, output_dir, image_dir):
    """
    将单个XML标注文件转换为YOLO格式的txt文件。
    :param xml_file_path: 原始XML文件的路径。
    :param output_dir: YOLO格式txt文件的输出目录。
    :param image_dir: 原始图像目录，用于获取图像尺寸。
    """
    try:
        # 解析XML文件
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # 获取文件名（不含后缀）
        base_name = os.path.splitext(os.path.basename(xml_file_path))[0]
        output_path = os.path.join(output_dir, base_name + '.txt')

        # 获取图像尺寸
        size = root.find('size')
        if size is None:
            # 尝试从图像文件获取尺寸
            image_path = os.path.join(image_dir, base_name + '.jpg')
            if not os.path.exists(image_path):
                print(f"Warning: Image for {base_name} not found. Skipping.")
                return
            
            from PIL import Image
            img = Image.open(image_path)
            width = img.width
            height = img.height
        else:
            width = int(size.find('width').text)
            height = int(size.find('height').text)

        # 写入YOLO格式的txt文件
        with open(output_path, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in classes:
                    print(f"Warning: Class '{class_name}' not in defined classes. Skipping.")
                    continue
                
                class_id = classes[class_name]
                bndbox = obj.find('bndbox')
                
                # 提取边界框坐标
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # 归一化处理
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                # 写入文件
                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
        print(f"Converted {xml_file_path} to YOLO format.")
    except Exception as e:
        print(f"Error processing {xml_file_path}: {e}")

def process_all_xmls(xml_dir, output_dir, image_dir):
    """
    处理指定目录下的所有XML文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml'):
            xml_path = os.path.join(xml_dir, filename)
            convert_xml_to_yolo(xml_path, output_dir, image_dir)

if __name__ == '__main__':
    # 请根据你的实际路径修改以下变量
    raw_xml_dir = 'data/raw/annotations/'
    yolo_labels_dir = 'data/raw/labels/'
    raw_images_dir = 'data/raw/images/'
    
    process_all_xmls(raw_xml_dir, yolo_labels_dir, raw_images_dir)
    print("All XML files have been processed.")