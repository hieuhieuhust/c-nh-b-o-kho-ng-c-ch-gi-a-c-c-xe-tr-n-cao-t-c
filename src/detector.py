from typing import List, Dict, Any
from ultralytics import YOLO
import numpy as np

class YOLOVehicleDetector:
    """
    Wrapper cho mô hình phát hiện phương tiện giao thông sử dụng YOLOv8 / YOLO11.
    """
    def __init__(self, model_path: str = "models/yolov8n.pt", 
                 vehicle_classes: Dict[int, Dict[str, Any]] = None, 
                 conf_thresh: float = 0.4,
                 img_size: int = 640):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.img_size = img_size
        self.vehicle_classes = vehicle_classes or {
            2: {"name": "O to", "real_height_m": 1.5},
            3: {"name": "Xe may", "real_height_m": 1.1},
            5: {"name": "Xe buyt", "real_height_m": 3.0},
            7: {"name": "Xe tai", "real_height_m": 2.8}
        }
        
        # Tải mô hình YOLO
        print(f"[Detector] Đang tải mô hình từ: {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.target_class_ids = list(self.vehicle_classes.keys())

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Thực hiện inference trên 1 frame ảnh và trả về danh sách các phương tiện phát hiện được.
        
        Trả về danh sách dict:
        [
            {
                "cls_id": int,
                "name": str,
                "box": (x1, y1, x2, y2),
                "conf": float,
                "real_height_m": float,
                "bbox_height": int,
                "center_x": int,
                "center_y": int
            }, ...
        ]
        """
        results = self.model(
            frame, 
            classes=self.target_class_ids, 
            conf=self.conf_thresh, 
            imgsz=self.img_size,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            veh_info = self.vehicle_classes.get(cls_id, {"name": "Phuong tien", "real_height_m": 1.5})
            bbox_h = max(1, y2 - y1)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "cls_id": cls_id,
                "name": veh_info["name"],
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "real_height_m": veh_info["real_height_m"],
                "bbox_height": bbox_h,
                "center_x": cx,
                "center_y": cy
            })

        return detections
