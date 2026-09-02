from collections import deque
from typing import Optional

class DistanceEstimator:
    """
    Ước lượng khoảng cách quang học dựa trên mô hình Pinhole Camera chuẩn hóa độ phân giải.
    Công thức: d = (Focal_Length_scaled * Real_Height) / BoundingBox_Height
    Tự động co giãn theo mọi độ phân giải camera (480p, 720p, 1080p, 4K).
    """
    def __init__(self, focal_length: float = 700.0, ref_height: float = 720.0):
        self.base_focal_length = focal_length
        self.ref_height = ref_height

    def calculate_distance(self, real_height_m: float, bbox_height_px: int, current_frame_height: Optional[int] = None) -> float:
        """
        Tính khoảng cách tính bằng mét từ camera đến xe.
        Tự động cân bằng tỉ lệ pixel khi đổi độ phân giải ảnh.
        """
        if bbox_height_px <= 0:
            return 999.0

        # Nếu có truyền chiều cao khung hình hiện tại, tự động co giãn tiêu cự tương ứng
        if current_frame_height is not None and current_frame_height > 0:
            scale_factor = current_frame_height / self.ref_height
            effective_focal = self.base_focal_length * scale_factor
        else:
            effective_focal = self.base_focal_length

        distance = (effective_focal * real_height_m) / float(bbox_height_px)
        return round(distance, 2)


class DistanceSmoother:
    """
    Bộ lọc làm mượt khoảng cách (Moving Average Filter) chống giật số giữa các frame.
    """
    def __init__(self, window_size: int = 5, max_jump_m: float = 15.0):
        self.window_size = window_size
        self.max_jump_m = max_jump_m
        self.history = deque(maxlen=window_size)

    def update(self, new_distance: float) -> float:
        """
        Cập nhật giá trị khoảng cách mới và trả về giá trị đã làm mượt.
        """
        if len(self.history) > 0:
            # Nếu khoảng cách nhảy đột ngột quá lớn (ví dụ chuyển xe khác), reset bộ đệm
            if abs(new_distance - self.history[-1]) > self.max_jump_m:
                self.history.clear()

        self.history.append(new_distance)
        smoothed = sum(self.history) / len(self.history)
        return round(smoothed, 2)

    def reset(self):
        """Reset bộ nhớ làm mượt khi không có xe mục tiêu."""
        self.history.clear()