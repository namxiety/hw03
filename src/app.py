import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.face_utils import (
    detect_and_encode,
    draw_face_boxes,
    compare_faces,
    load_known_faces
)

# 页面配置
st.set_page_config(page_title="人脸检测与识别", layout="wide")
st.title("🔍 人脸检测与识别系统")

# 加载已知人脸库
known_faces = load_known_faces()

# 上传图片或选择示例图
option = st.radio("选择图片来源", ("上传图片", "使用示例图"))

if option == "上传图片":
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
elif option == "使用示例图":
    # 示例图路径
    example_path = "examples/example.jpg"
    image = Image.open(example_path)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# 处理图片
if 'image_bgr' in locals():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图片")
        st.image(image, use_column_width=True)
    
    # 检测人脸
    face_locations, face_encodings = detect_and_encode(image_bgr)
    
    # 比对识别
    names = []
    for encoding in face_encodings:
        name = compare_faces(known_faces, encoding)
        names.append(name)
    
    # 绘制结果
    result_image = draw_face_boxes(image_bgr, face_locations, names)
    
    with col2:
        st.subheader("检测结果")
        st.image(result_image, use_column_width=True)
    
    # 显示编码信息
    with st.expander("人脸128维特征编码"):
        for i, encoding in enumerate(face_encodings):
            st.write(f"人脸 {i+1} 编码：")
            st.write(encoding)
