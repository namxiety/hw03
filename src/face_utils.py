import face_recognition
import cv2
import numpy as np
from PIL import Image

def detect_and_encode(image: np.ndarray):
    """
    检测人脸并生成128维特征编码
    返回：人脸位置列表、人脸编码列表
    """
    # 转换为RGB格式（face_recognition要求）
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测人脸位置
    face_locations = face_recognition.face_locations(rgb_image)
    # 生成人脸特征编码
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    return face_locations, face_encodings

def draw_face_boxes(image: np.ndarray, face_locations: list, names: list = None):
    """
    在图片上框选人脸并标注姓名（若有）
    """
    img_copy = image.copy()
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # 绘制矩形框
        cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 255, 0), 2)
        # 标注姓名（如果有）
        if names and i < len(names):
            cv2.putText(img_copy, names[i], (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

def compare_faces(known_encodings: list, unknown_encoding: np.ndarray, tolerance: float = 0.6):
    """
    与已知人脸库比对，返回匹配的姓名
    """
    if not known_encodings:
        return "Unknown"
    
    matches = face_recognition.compare_faces([enc[1] for enc in known_encodings], unknown_encoding, tolerance=tolerance)
    face_distances = face_recognition.face_distance([enc[1] for enc in known_encodings], unknown_encoding)
    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        return known_encodings[best_match_index][0]
    return "Unknown"

# 初始化已知人脸库（示例）
def load_known_faces():
    """
    加载已知人脸库，格式：[(姓名, 编码), ...]
    """
    # 这里可以替换为自己的已知人脸图片
    known_faces = []
    # 示例：如果有示例图片可以在这里加载
    # obama_image = face_recognition.load_image_file("examples/obama.jpg")
    # obama_encoding = face_recognition.face_encodings(obama_image)[0]
    # known_faces.append(("Obama", obama_encoding))
    return known_faces
