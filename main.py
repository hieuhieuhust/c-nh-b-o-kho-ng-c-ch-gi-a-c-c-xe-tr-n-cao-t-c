import argparse
import time
import os
import cv2
import yaml

from src.detector import YOLOVehicleDetector
from src.distance_estimator import DistanceEstimator, DistanceSmoother
from src.roi_filter import EgoCorridorFilter
from src.voice_alert import VoiceAlertManager
from src.visualizer import Visualizer
from src.tracker import VehicleTracker

# Biến toàn cục lưu sự kiện click chuột
mouse_click_pt = None

def on_mouse_click(event, x, y, flags, param):
    global mouse_click_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pt = (x, y)

def load_config(config_path: str) -> dict:
    """Đọc file cấu hình YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    global mouse_click_pt
    parser = argparse.ArgumentParser(description="Hệ thống đo khoảng cách xe đa chế độ: Vùng Xanh Rảnh Tay & Chạm Chọn Xe")
    parser.add_argument("--source", type=str, default="0", 
                        help="Đường dẫn file video (ví dụ: data/raw/cao_toc_phap_van_1.mp4) hoặc '0' cho Webcam")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", 
                        help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--calibrate", action="store_true", 
                        help="Bật chế độ căn chỉnh tương tác kéo thanh trượt vùng xanh")
    parser.add_argument("--save-output", type=str, default=None, 
                        help="Đường dẫn lưu video kết quả (ví dụ: data/output/result.mp4)")
    parser.add_argument("--no-tts", action="store_true", 
                        help="Tắt hoàn toàn âm thanh giọng nói")
    parser.add_argument("--headless", action="store_true", 
                        help="Chạy ẩn không mở cửa sổ GUI")
    args = parser.parse_args()

    # 1. Nếu người dùng chọn chế độ căn chỉnh tương tác
    if args.calibrate:
        from scripts.calibrate_roi import main as run_calibration
        run_calibration()
        return

    # 2. Tải cấu hình
    cfg = load_config(args.config)
    cam_cfg = cfg.get("camera", {})
    corridor_cfg = cfg.get("corridor", {})
    veh_classes = cfg.get("vehicle_classes", {})
    alert_cfg = cfg.get("alert_thresholds", {})
    smooth_cfg = cfg.get("smoothing", {})
    infer_cfg = cfg.get("inference", {})

    # 3. Khởi tạo các module
    detector = YOLOVehicleDetector(
        model_path=infer_cfg.get("model_path", "models/yolov8n.pt"),
        vehicle_classes=veh_classes,
        conf_thresh=infer_cfg.get("confidence_threshold", 0.35),
        img_size=infer_cfg.get("img_size", 640)
    )

    distance_estimator = DistanceEstimator(
        focal_length=cam_cfg.get("focal_length", 700.0)
    )

    smoother = DistanceSmoother(
        window_size=smooth_cfg.get("window_size", 5)
    )

    corridor_filter = EgoCorridorFilter(
        top_y_ratio=corridor_cfg.get("top_y_ratio", 0.58),
        top_width_ratio=corridor_cfg.get("top_width_ratio", 0.14),
        bot_width_ratio=corridor_cfg.get("bot_width_ratio", 0.44),
        center_x_ratio=corridor_cfg.get("center_x_ratio", 0.50)
    )

    tracker = VehicleTracker()

    tts_enabled = alert_cfg.get("tts_enabled", True) and (not args.no_tts)
    voice_alert = VoiceAlertManager(
        cooldown_sec=alert_cfg.get("tts_cooldown_sec", 2.5),
        danger_dist=alert_cfg.get("danger_distance_m", 5.0),
        warning_dist=alert_cfg.get("warning_distance_m", 10.0),
        enabled=tts_enabled
    )

    visualizer = Visualizer(
        danger_dist=alert_cfg.get("danger_distance_m", 5.0),
        warning_dist=alert_cfg.get("warning_distance_m", 10.0)
    )

    # 4. Mở nguồn Video
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Lỗi] Không thể mở nguồn video: {args.source}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    video_writer = None
    if args.save_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save_output, fourcc, fps_in, (frame_w, frame_h))
        print(f"[Record] Sẽ lưu video kết quả tại: {args.save_output}")

    corridor_poly, left_pts, right_pts = corridor_filter.get_corridor_polygon(frame_w, frame_h)

    # Cửa sổ hiển thị
    window_name = "Vehicle Distance Alert - Dual Mode System"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse_click)

    print("\n========================================================")
    print("▶ HỆ THỐNG ĐÃ SẴN SÀNG (HỖ TRỢ 2 CHẾ ĐỘ THÔNG MINH)!")
    print("👉 Chế độ 1 (Mặc định): VÙNG XANH RẢNH TAY (Tự động lọc xe cùng làn).")
    print("👉 Chế độ 2: CHẠM CHỌN XE THỦ CÔNG (Click chuột vào bất kỳ xe nào để khóa).")
    print("👉 Bấm phím 'm' trên bàn phím để ĐỔI QUA LẠI 2 CHẾ ĐỘ.")
    print("👉 Bấm phím 's' để Bật/Tắt âm thanh cảnh báo.")
    print("👉 Bấm phím 'q' để dừng chương trình.")
    print("========================================================\n")

    # Mặc định là Chế độ Vùng xanh rảnh tay (False = Auto, True = Manual Tap-to-Track)
    is_manual_mode = False

    prev_time = time.time()
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"[Info] Đã xử lý xong toàn bộ video ({frame_idx} frames).")
                break

            frame_idx += 1

            # Đo FPS
            curr_time = time.time()
            fps = 1.0 / max(0.001, (curr_time - prev_time))
            prev_time = curr_time

            # 5. Bước 1: Nhận diện tất cả các xe (YOLOv8)
            detections = detector.detect(frame)

            target_vehicle = None
            mode_label = "RANH TAY"

            # 6. Bước 2: Xử lý theo chế độ hiện tại
            if is_manual_mode:
                mode_label = "CHON TAY"
                # Nếu người dùng vừa click chuột lên màn hình
                if mouse_click_pt is not None:
                    cx_click, cy_click = mouse_click_pt
                    mouse_click_pt = None
                    locked = tracker.lock_at_point(cx_click, cy_click, detections)
                    if locked:
                        print(f"🎯 [ĐÃ KHÓA XE THỦ CÔNG]: {locked['name']} tại ({cx_click}, {cy_click})")
                    else:
                        print("❌ Đã hủy khóa xe.")

                # Cập nhật vị trí bám đuôi xe mục tiêu
                target_vehicle = tracker.update(detections)

                # Vẽ tất cả các xe còn lại dạng viền mỏng kèm số thứ tự
                selected_idx = None
                if target_vehicle and target_vehicle in detections:
                    selected_idx = detections.index(target_vehicle)
                frame = visualizer.draw_all_candidates(frame, detections, selected_idx)

            else:
                # CHẾ ĐỘ TỰ ĐỘNG: Lọc theo vùng xanh rảnh tay
                target_vehicle, ignored_vehicles = corridor_filter.filter_vehicles(detections, frame_w, frame_h)
                is_locked = (target_vehicle is not None)
                frame = visualizer.draw_ego_corridor(frame, corridor_poly, left_pts, right_pts, is_locked=is_locked)
                frame = visualizer.draw_ignored_vehicles(frame, ignored_vehicles)

            target_info_hud = None

            # 7. Bước 3: Xử lý xe mục tiêu đang được theo dõi
            if target_vehicle:
                raw_distance = distance_estimator.calculate_distance(
                    target_vehicle["real_height_m"],
                    target_vehicle["bbox_height"],
                    current_frame_height=frame_h
                )
                smoothed_dist = smoother.update(raw_distance)

                # Vẽ xe mục tiêu
                frame = visualizer.draw_target_vehicle(frame, target_vehicle, smoothed_dist, mode_label)
                target_info_hud = (target_vehicle["name"], smoothed_dist)

                # Cảnh báo giọng nói
                voice_alert.notify(target_vehicle["name"], smoothed_dist)
            else:
                smoother.reset()

            # 8. Bước 4: Vẽ thanh Dashboard HUD
            frame = visualizer.draw_hud(frame, fps, target_info_hud, voice_alert.enabled, is_manual_mode)

            # In tiến trình mỗi 60 frame
            if frame_idx % 60 == 0 or frame_idx == 1:
                target_str = f"{target_info_hud[0]} ({target_info_hud[1]}m)" if target_info_hud else "Chưa chọn xe"
                mode_str = "CHỌN TAY" if is_manual_mode else "RẢNH TAY"
                progress_pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"[Frame {frame_idx}/{total_frames} ({progress_pct:.1f}%)] [{mode_str}] Mục tiêu: {target_str} | FPS: {fps:.1f}")

            # 9. Ghi video / Hiển thị màn hình
            if video_writer:
                video_writer.write(frame)

            if not args.headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[User] Người dùng bấm 'q' để thoát.")
                    break
                elif key == ord('s'):
                    voice_alert.toggle()
                elif key == ord('m'):
                    is_manual_mode = not is_manual_mode
                    tracker.unlock()
                    status_m = "CHẠM CHỌN XE THỦ CÔNG (Click vào xe)" if is_manual_mode else "VÙNG XANH RẢNH TAY (Tự động)"
                    print(f"\n🔄 [ĐÃ ĐỔI CHẾ ĐỘ SANG]: {status_m}\n")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        voice_alert.stop()
        cv2.destroyAllWindows()
        print("[Hoàn tất] Đã giải phóng tài nguyên an toàn.")

if __name__ == "__main__":
    main()
