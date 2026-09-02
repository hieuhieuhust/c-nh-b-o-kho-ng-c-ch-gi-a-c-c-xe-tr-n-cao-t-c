import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class EgoCorridorFilter:
    """
    Hệ thống phân tích làn đường theo Hành Lang Định Vị Kính Lái (Calibrated Ego-Corridor).
    Không phụ thuộc vào vạch sơn mờ, chống rung giật 100% và cực kỳ chính xác.
    """
    def __init__(self, 
                 top_y_ratio: float = 0.56, 
                 top_width_ratio: float = 0.22, 
                 bot_width_ratio: float = 0.65,
                 center_x_ratio: float = 0.50):
        self.top_y_ratio = top_y_ratio
        self.top_w_ratio = top_width_ratio
        self.bot_w_ratio = bot_width_ratio
        self.cx_ratio = center_x_ratio

    def get_corridor_polygon(self, frame_width: int, frame_height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tạo đa giác Hành lang làn đường cố định theo phối cảnh chuẩn của kính lái.
        Trả về: (corridor_polygon, left_line_pts, right_line_pts)
        """
        w, h = frame_width, frame_height
        cx = int(w * self.cx_ratio)
        top_y = int(h * self.top_y_ratio)
        bot_y = h - 1

        top_w = int(w * self.top_w_ratio)
        bot_w = int(w * self.bot_w_ratio)

        # 4 đỉnh của hình thang làn đường chuẩn
        top_left = (max(0, cx - top_w // 2), top_y)
        top_right = (min(w - 1, cx + top_w // 2), top_y)
        bot_right = (min(w - 1, cx + bot_w // 2), bot_y)
        bot_left = (max(0, cx - bot_w // 2), bot_y)

        polygon = np.array([top_left, top_right, bot_right, bot_left], dtype=np.int32)
        left_pts = np.array([top_left, bot_left], dtype=np.int32)
        right_pts = np.array([top_right, bot_right], dtype=np.int32)

        return polygon, left_pts, right_pts

    def filter_vehicles(self, detections: List[Dict[str, Any]], frame_width: int, frame_height: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Lọc những xe có điểm bánh xe tiếp đất nằm lọt trong Hành lang làn đường xe mình.
        """
        polygon, _, _ = self.get_corridor_polygon(frame_width, frame_height)

        in_corridor_vehicles = []
        ignored_vehicles = []

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            bx = (x1 + x2) // 2
            by = y2
            det["bottom_point"] = (bx, by)

            # Kiểm tra xem bánh xe tiếp đất có nằm trong Hành lang làn đường không
            is_inside = cv2.pointPolygonTest(polygon, (float(bx), float(by)), False) >= 0

            if is_inside:
                in_corridor_vehicles.append(det)
            else:
                ignored_vehicles.append(det)

        if not in_corridor_vehicles:
            return None, ignored_vehicles

        # Chọn xe gần nhất trong hành lang (đáy y2 lớn nhất)
        target_vehicle = max(in_corridor_vehicles, key=lambda d: d["box"][3])

        for veh in in_corridor_vehicles:
            if veh is not target_vehicle:
                ignored_vehicles.append(veh)

        return target_vehicle, ignored_vehicles
