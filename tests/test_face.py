
---

## 六、测试用例（tests/test_face.py，可选）
```python
import cv2
import numpy as np
from src.face_utils import detect_and_encode

def test_detect_face():
    # 创建一张空白测试图片
    img = np.zeros((500, 500, 3), np.uint8)
    # 检测人脸（无人脸）
    locs, encs = detect_and_encode(img)
    assert len(locs) == 0
    assert len(encs) == 0
