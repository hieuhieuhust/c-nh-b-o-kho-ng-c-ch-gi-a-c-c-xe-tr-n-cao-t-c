import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class Visualizer:
    """
    Module giao diện HUD hỗ trợ cả 2 CHẾ ĐỘ:
    1. MODE AUTO: Tự động lọc xe theo vùng xanh rảnh tay (Auto Ego-Corridor).
    2. MODE MANUAL: Chạm tay chọn bất kỳ chiếc xe nào trên màn hình để khóa (Tap-to-Lock Tracking).
    """
    def __init__(self, danger_dist: float = 5.0, warning_dist: float = 10.0):
        self.danger_dist = danger_dist
        self.warning_dist = warning_dist

        # Bảng màu BGR
        self.COLOR_DANGER = (0, 0, 255)         # Đỏ
        self.COLOR_WARNING = (0, 165, 255)      # Cam
        self.COLOR_SAFE = (0, 255, 0)           # Xanh lá
        self.COLOR_LOCKED = (0, 255, 255)       # Vàng rực cho xe được người dùng chạm chọn
        self.COLOR_CORRIDOR_LINE = (255, 200, 0)# Vàng Neon viền hành lang
        self.COLOR_CORRIDOR_FILL = (0, 230, 120)# Thảm xanh mờ
        self.COLOR_CANDIDATE = (200, 200, 200)  # Viền trắng nhạt cho các xe có thể chạm chọn

    def draw_ego_corridor(self, frame: np.ndarray, polygon: np.ndarray, left_pts: np.ndarray, right_pts: np.ndarray, is_locked: bool = False) -> np.ndarray:
        """Vẽ hành lang vùng xanh rảnh tay trong chế độ TỰ ĐỘNG."""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], self.COLOR_CORRIDOR_FILL)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        line_color = (0, 255, 255) if is_locked else self.COLOR_CORRIDOR_LINE
        cv2.polylines(frame, [left_pts], False, line_color, 2, cv2.LINE_AA)
        cv2.polylines(frame, [right_pts], False, line_color, 2, cv2.LINE_AA)
        return frame

    def draw_ignored_vehicles(self, frame: np.ndarray, ignored_vehicles: List[Dict[str, Any]]) -> np.ndarray:
        """Vẽ các xe ở làn khác (làn trái/phải) màu xám mờ không quan trọng."""
        for veh in ignored_vehicles:
            x1, y1, x2, y2 = veh["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
        return frame

    def draw_all_candidates(self, frame: np.ndarray, detections: List[Dict[str, Any]], selected_idx: Optional[int]) -> np.ndarray:
        """
        Trong chế độ CHẠM CHỌN XE THỦ CÔNG:
        Vẽ tất cả các xe phát hiện được trên đường với nút bấm chạm để người dùng dễ dàng chọn.
        """
        for i, det in enumerate(detections):
            if i == selected_idx:
                continue # Xe được chọn sẽ vẽ riêng nổi bật
            x1, y1, x2, y2 = det["box"]
            name = det["name"]

            # Vẽ khung chữ nhật mờ
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_CANDIDATE, 1)
            # Vẽ nhãn nhỏ "Chạm để khóa"
            lbl = f"[{i+1}] {name}"
            cv2.putText(frame, lbl, (x1 + 3, max(15, y1 - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_CANDIDATE, 1, cv2.LINE_AA)
        return frame

    def draw_target_vehicle(self, frame: np.ndarray, target: Dict[str, Any], distance: float, mode_str: str = "RANH TAY") -> np.ndarray:
        """
        Vẽ nổi bật chiếc xe đang được KHÓA MỤC TIÊU và hiển thị khoảng cách.
        """
        x1, y1, x2, y2 = target["box"]
        name = target["name"]
        bx, by = (x1 + x2) // 2, y2

        # Màu cảnh báo theo mét
        if distance < self.danger_dist:
            color = self.COLOR_DANGER
            status_text = "NGUY HIEM - PHANH GAP!"
        elif distance < self.warning_dist:
            color = self.COLOR_WARNING
            status_text = "CHU Y GIAM TOC"
        else:
            color = self.COLOR_SAFE
            status_text = "AN TOAN"

        # 1. Đánh dấu điểm tiếp đất bánh xe
        cv2.circle(frame, (bx, by), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 12, color, 2, cv2.LINE_AA)

        # 2. Bounding Box xe mục tiêu
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # 3. Vẽ 4 góc Radar
        line_len = min(25, (x2 - x1) // 4)
        for (cx, cy), (dx, dy) in [
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * line_len, cy), color, 4)
            cv2.line(frame, (cx, cy), (cx, cy + dy * line_len), color, 4)

        # 4. Nhãn khoảng cách
        label = f"[{mode_str}] {name.upper()}: {distance:.1f}m ({status_text})"
        (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        top_y = max(lbl_h + 10, y1 - 10)

        cv2.rectangle(frame, (x1, top_y - lbl_h - 6), (x1 + lbl_w + 10, top_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 5, top_y - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def draw_hud(self, frame: np.ndarray, fps: float, target_info: Optional[Tuple[str, float]], tts_enabled: bool, is_manual_mode: bool = False) -> np.ndarray:
        """Vẽ Dashboard thanh trên cùng hỗ trợ chuyển đổi 2 chế độ."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)

        tts_str = "[LOA: BAT]" if tts_enabled else "[LOA: TAT]"
        mode_tag = "[CHẾ ĐỘ: CHẠM CHỌN XE]" if is_manual_mode else "[CHẾ ĐỘ: VÙNG XANH RẢNH TAY]"
        mode_color = (0, 255, 255) if is_manual_mode else (0, 255, 120)

        cv2.putText(frame, f"FPS: {fps:.1f} | {tts_str} | {mode_tag}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)
        
        hint = "Phim: 'M' (Doi che do) | Click chuot (Chon xe) | 'S' (Loa) | 'Q' (Thoat)"
        cv2.putText(frame, hint, (20, 52), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        if target_info:
            name, dist = target_info
            color = self.COLOR_DANGER if dist < self.danger_dist else (
                self.COLOR_WARNING if dist < self.warning_dist else self.COLOR_SAFE
            )
            hud_text = f"XE DANG THEO DOI: {dist:.1f}m ({name.upper()})"
            cv2.putText(frame, hud_text, (w // 2 - 240, 42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 3, cv2.LINE_AA)
        else:
            txt = "CLICK CHON 1 XE BAT KY DE DO" if is_manual_mode else "VUNG XANH TRONG (KHONG CO XE)"
            cv2.putText(frame, txt, (w // 2 - 230, 42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (120, 120, 120), 2, cv2.LINE_AA)

        return frame
