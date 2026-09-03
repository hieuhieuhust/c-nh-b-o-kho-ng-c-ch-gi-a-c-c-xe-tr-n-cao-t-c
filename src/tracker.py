import math
from typing import List, Dict, Any, Optional

class VehicleTracker:
    """
    Bộ theo dõi mục tiêu (Target Lock Tracker) sử dụng khoảng cách tâm và IoU.
    Giúp duy trì việc bám đuôi chính xác 1 chiếc xe cụ thể khi người dùng chạm tay chọn.
    """
    def __init__(self, max_distance_thresh: float = 120.0):
        self.max_distance_thresh = max_distance_thresh
        self.locked_vehicle = None

    def lock_at_point(self, click_x: int, click_y: int, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Khóa chiếc xe chứa tọa độ click chuột (x, y) của người dùng.
        """
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.locked_vehicle = det
                return det

        # Nếu không click trúng trong bounding box, tìm xe có tâm gần nhất
        closest = None
        min_dist = 60.0 # Bán kính tìm kiếm lân cận
        for det in detections:
            cx, cy = det["center_x"], det["center_y"]
            dist = math.hypot(cx - click_x, cy - click_y)
            if dist < min_dist:
                min_dist = dist
                closest = det

        self.locked_vehicle = closest
        return closest

    def unlock(self):
        """Hủy khóa mục tiêu."""
        self.locked_vehicle = None

    def update(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Cập nhật vị trí mới của chiếc xe đang bị khóa ở frame tiếp theo.
        """
        if self.locked_vehicle is None or not detections:
            self.locked_vehicle = None
            return None

        prev_cx = self.locked_vehicle["center_x"]
        prev_cy = self.locked_vehicle["center_y"]
        prev_h = self.locked_vehicle["bbox_height"]

        best_match = None
        min_cost = float("inf")

        for det in detections:
            cx, cy = det["center_x"], det["center_y"]
            dist = math.hypot(cx - prev_cx, cy - prev_cy)

            # Độ chênh lệch chiều cao bbox
            h_diff = abs(det["bbox_height"] - prev_h)

            cost = dist + h_diff * 0.5

            if dist < self.max_distance_thresh and cost < min_cost:
                min_cost = cost
                best_match = det

        self.locked_vehicle = best_match
        return best_match
