import cv2
import threading
import numpy as np
import time
import datetime
import os
from pyorbbecsdk import Context, Config, OBError, OBSensorType, Pipeline, FrameSet, OBFormat

ESC_KEY = 27


class CameraThread(threading.Thread):
    # 保持原有CameraThread类实现不变
    def __init__(self, device, device_idx, image_list, lock):
        super().__init__()
        self.device = device
        self.device_idx = device_idx
        self.image_list = image_list
        self.lock = lock
        self.running = True
        self.last_valid_frame = None
        self.actual_width = None
        self.actual_height = None
        self.actual_fps = None

    def run(self):
        config = Config()
        pipeline = None
        try:
            pipeline = Pipeline(self.device)
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = self.select_best_profile(profile_list)

            if not color_profile:
                raise RuntimeError("No suitable video stream configuration found")

            print(f"\nCamera {self.device_idx} Config: "
                  f"{self.actual_width}x{self.actual_height} @{self.actual_fps}fps")

            config.enable_stream(color_profile)
            pipeline.start(config)
            print(f"Camera {self.device_idx} started successfully")

            while self.running:
                try:
                    frames = pipeline.wait_for_frames(500)
                    if not frames:
                        continue

                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    image = self.process_frame(color_frame)
                    with self.lock:
                        if image is not None:
                            self.image_list[self.device_idx] = image.copy()
                            self.last_valid_frame = image.copy()

                except Exception as e:
                    print(f"Camera {self.device_idx} error: {str(e)}")
                    break

        except Exception as e:
            print(f"\nCamera {self.device_idx} init failed: {str(e)}")
            self.print_available_profiles(profile_list)
        finally:
            if pipeline:
                pipeline.stop()
            print(f"Camera {self.device_idx} stopped")

    def select_best_profile(self, profile_list):
        format_candidates = [
            getattr(OBFormat, "RGB", None),
            getattr(OBFormat, "OB_FORMAT_RGB", None),
        ]
        valid_formats = [fmt for fmt in format_candidates if fmt is not None]

        target_config = (1920, 1080, 30)
        for fmt in valid_formats:
            try:
                profile = profile_list.get_video_stream_profile(
                    target_config[0], target_config[1], fmt, target_config[2]
                )
                self.actual_width = target_config[0]
                self.actual_height = target_config[1]
                self.actual_fps = target_config[2]
                return profile
            except (OBError, AttributeError):
                continue

        print(f"Camera {self.device_idx} cannot find 1080p30 config, using default")
        try:
            default_profile = profile_list.get_default_video_stream_profile()
            self.actual_width = default_profile.get_width()
            self.actual_height = default_profile.get_height()
            self.actual_fps = default_profile.get_fps()
            return default_profile
        except Exception as e:
            print(f"Get default profile failed: {str(e)}")
            return None

    def process_frame(self, frame):
        try:
            fmt = frame.get_format()
            data = np.frombuffer(frame.get_data(), dtype=np.uint8)
            h, w = frame.get_height(), frame.get_width()

            if fmt == OBFormat.RGB:
                rgb_image = data.reshape((h, w, 3))
                # 确保数组是C连续的
                if not rgb_image.flags.c_contiguous:
                    rgb_image = np.ascontiguousarray(rgb_image)
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            elif fmt in [OBFormat.YUYV, OBFormat.YUY2]:
                yuyv_image = data.reshape((h, w, 2))
                # 确保数组是C连续的
                if not yuyv_image.flags.c_contiguous:
                    yuyv_image = np.ascontiguousarray(yuyv_image)
                return cv2.cvtColor(yuyv_image, cv2.COLOR_YUV2BGR_YUYV)
            elif fmt == OBFormat.MJPG:
                # 对于MJPG格式，使用imdecode后也需要确保连续性
                decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if decoded is not None and not decoded.flags.c_contiguous:
                    return np.ascontiguousarray(decoded)
                return decoded
            else:
                raise ValueError(f"Unsupported format: {fmt.name}")
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            error_img = np.zeros((self.actual_height or 1080, self.actual_width or 1920, 3), dtype=np.uint8)
            cv2.putText(error_img, f"Error: {str(e)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_img

    def print_available_profiles(self, profile_list):
        print(f"\nCamera {self.device_idx} Available Profiles:")
        try:
            for p in profile_list.get_all_video_stream_profiles():
                print(f"  {p.get_width()}x{p.get_height()} {p.get_format().name}@{p.get_fps()}fps")
        except Exception as e:
            print(f"Get profiles failed: {str(e)}")


def main():
    context = Context()
    device_list = context.query_devices()

    if device_list.get_count() < 3:
        print(f"Error: Need 3 cameras (found {device_list.get_count()})")
        return

    devices = []
    for i in range(3):
        try:
            serial = device_list.get_device_serial_number_by_index(i)
            devices.append(device_list.get_device_by_serial_number(serial))
            print(f"Camera {i} connected (SN: {serial})")
        except Exception as e:
            print(f"Camera {i} init failed: {str(e)}")
            return

    image_list = [None] * 3
    lock = threading.Lock()
    threads = []
    for i in range(3):
        thread = CameraThread(devices[i], i, image_list, lock)
        thread.start()
        threads.append(thread)

    # 初始化计数器
    photo_counts = {
        'cam0': 0,
        'cam1': 0,
        'cam2': 0,
        'simultaneous': 0
    }

    # Recording variables
    is_recording = False
    video_writers = [None] * 3
    recording_start_time = 0

    try:
        cv2.namedWindow("Multi Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi Camera", 1920, 720)

        while True:
            with lock:
                current_frames = [img.copy() if img is not None else
                                  np.zeros((threads[i].actual_height or 1080,
                                            threads[i].actual_width or 1920, 3), dtype=np.uint8)
                                  for i, img in enumerate(image_list)]

            # Resize for display
            display_frames = []
            for f in current_frames:
                # 确保显示帧也是C连续的
                if f is not None and not f.flags.c_contiguous:
                    f = np.ascontiguousarray(f)
                resized = cv2.resize(f, (640, 360))
                display_frames.append(resized)

            combined = cv2.hconcat(display_frames)

            # 添加计数显示
            count_text = (f"Cam1: {photo_counts['cam0']}  "
                          f"Cam2: {photo_counts['cam1']}  "
                          f"Cam3: {photo_counts['cam2']}  "
                          f"Group: {photo_counts['simultaneous']}")
            cv2.putText(combined, count_text, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Handle recording
            if is_recording:
                elapsed = time.time() - recording_start_time
                time_str = f"Recording {datetime.timedelta(seconds=int(elapsed))}"
                cv2.putText(combined, time_str, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Write original frames
                for i in range(3):
                    if video_writers[i] is not None:
                        # 确保写入的帧是C连续的
                        frame_to_write = current_frames[i]
                        if not frame_to_write.flags.c_contiguous:
                            frame_to_write = np.ascontiguousarray(frame_to_write)
                        video_writers[i].write(frame_to_write)

            cv2.imshow("Multi Camera", combined)

            key = cv2.waitKey(1)
            if key in (ord('q'), ESC_KEY):
                break
            elif key == ord('r'):
                if not is_recording:
                    try:
                        # Verify camera configs
                        for i in range(3):
                            if (threads[i].actual_width != 1920 or
                                    threads[i].actual_height != 1080 or
                                    threads[i].actual_fps != 30):
                                raise ValueError(f"Camera {i} config mismatch: "
                                                 f"{threads[i].actual_width}x{threads[i].actual_height}@{threads[i].actual_fps}fps")

                        # Init writers
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        os.makedirs("recordings", exist_ok=True)
                        for i in range(3):
                            filename = f"recordings/cam_{i}_{timestamp}.mp4"
                            video_writers[i] = cv2.VideoWriter(
                                filename, cv2.VideoWriter_fourcc(*'avc1'), 30, (1920, 1080)
                            )
                            if not video_writers[i].isOpened():
                                raise RuntimeError(f"Failed to create {filename}")

                        is_recording = True
                        recording_start_time = time.time()
                        print("Recording started")
                    except Exception as e:
                        print(f"Start recording failed: {str(e)}")
                        video_writers = [None] * 3
                else:
                    # Stop recording
                    for i in range(3):
                        if video_writers[i] is not None:
                            video_writers[i].release()
                    video_writers = [None] * 3
                    is_recording = False
                    print("Recording saved to 'recordings' folder")
            # 修改后的保存逻辑
            elif key == ord('s'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_sync_dir = os.path.join("calib_images", "sync")
                os.makedirs(base_sync_dir, exist_ok=True)
                success_count = 0

                for i in range(3):
                    if current_frames[i] is not None:
                        try:
                            cam_sync_dir = os.path.join(base_sync_dir, f"cam{i}")
                            os.makedirs(cam_sync_dir, exist_ok=True)
                            filename = os.path.join(cam_sync_dir, f"sync_{timestamp}.png")
                            # 确保保存的帧是C连续的
                            frame_to_save = current_frames[i]
                            if not frame_to_save.flags.c_contiguous:
                                frame_to_save = np.ascontiguousarray(frame_to_save)
                            if cv2.imwrite(filename, frame_to_save):
                                success_count += 1
                            else:
                                print(f"Failed to save cam{i}")
                        except Exception as e:
                            print(f"Save failed for cam{i}: {str(e)}")

                if success_count == 3:
                    photo_counts['simultaneous'] += 1
                    print(f"Group saved: {success_count}/3")
                else:
                    print(f"Partial group ({success_count}/3)")
            elif key in (ord('1'), ord('2'), ord('3')):
                camera_idx = key - ord('1')
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_dir = os.path.join("calib_images", "individual_snapshots", f"cam{camera_idx}")
                os.makedirs(save_dir, exist_ok=True)

                with lock:
                    frame = image_list[camera_idx].copy() if image_list[camera_idx] is not None else None

                if frame is not None:
                    try:
                        filename = os.path.join(save_dir, f"snapshot_{timestamp}.png")
                        # 确保保存的帧是C连续的
                        if not frame.flags.c_contiguous:
                            frame = np.ascontiguousarray(frame)
                        if cv2.imwrite(filename, frame):
                            photo_counts[f'cam{camera_idx}'] += 1
                            print(f"Saved camera {camera_idx} ({photo_counts[f'cam{camera_idx}']})")
                        else:
                            print(f"Save failed for camera {camera_idx}")
                    except Exception as e:
                        print(f"Save error for camera {camera_idx}: {str(e)}")
                else:
                    print(f"Camera {camera_idx} has no frame to save")

    finally:
        for thread in threads:
            thread.running = False
        for thread in threads:
            thread.join(timeout=1)
        for writer in video_writers:
            if writer is not None:
                writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()