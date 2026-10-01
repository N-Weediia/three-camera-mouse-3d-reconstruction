import cv2
import numpy as np
import os
import json
import re
import traceback
from pathlib import Path
from collections import defaultdict


# ================== 配置参数 ==================
class CheckerboardConfig:
    # 棋盘格参数
    PATTERN_SIZE = (11, 8)  # 内部角点数量 (cols-1, rows-1)
    SQUARE_SIZE = 0.035  # 单位：米

    # 存储路径配置（适配cam0目录结构）
    SINGLE_CALIB_DIR = "calib_images/individual_snapshots"  # 单目标定图像
    SYNC_CALIB_DIR = "calib_images/sync"  # 同步标定图像
    OUTPUT_DIR = "calibration_results"  # 标定结果

    # 标定参数
    MIN_SYNC_FRAMES = 10  # 最小同步帧数
    MIN_VALID_IMAGES = 10  # 单相机最小有效图像数
    COORD_SCALE = 3.0  # 坐标系显示比例

    @property
    def camera_paths(self):
        """生成相机路径配置（cam0, cam1, cam2目录）"""
        return {
            f"camera{i}": str(Path(self.SYNC_CALIB_DIR) / f"cam{i}")
            for i in range(3)
        }


# ================== 棋盘格标定类 ==================
class CheckerboardCalibrator:
    def __init__(self, config):
        self.config = config
        self.objp = self._generate_object_points()
        self.calib_data = {}
        self.sync_groups = self._load_sync_groups()

    def _generate_object_points(self):
        """生成棋盘格3D坐标点"""
        objp = np.zeros((np.prod(self.config.PATTERN_SIZE), 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.config.PATTERN_SIZE[0],
                      0:self.config.PATTERN_SIZE[1]].T.reshape(-1, 2)
        return objp * self.config.SQUARE_SIZE

    def _parse_image_info(self, filename):
        """解析文件名（适配cam_0_20250321_174436格式）"""
        try:
            stem = Path(filename).stem  # 移除扩展名
            # 匹配示例: cam_0_20240521_153045
            match = re.match(r"cam_(\d+)_(\d{8}_\d{6})", stem)
            if match:
                return {
                    "cam_idx": match.group(1),
                    "timestamp": match.group(2)
                }
            return None
        except Exception as e:
            print(f"文件名解析错误: {filename} - {str(e)}")
            return None

    def _load_sync_groups(self):
        """加载同步标定图像组（修复路径问题）"""
        sync_groups = defaultdict(dict)

        # 遍历所有相机目录
        for cam_id, cam_path in self.config.camera_paths.items():
            cam_dir = Path(cam_path)
            if not cam_dir.exists():
                raise FileNotFoundError(f"同步标定目录不存在: {cam_dir}")

            # 处理每个相机的同步图像
            for img_path in cam_dir.glob("*.png"):
                info = self._parse_image_info(img_path.name)
                if info:
                    timestamp = info["timestamp"]
                    sync_groups[timestamp][cam_id] = str(img_path)

        # 筛选有效同步组
        valid_groups = []
        for ts, cams in sync_groups.items():
            if len(cams) == 3:  # 必须包含三个相机的图像
                valid_groups.append(cams)

        if len(valid_groups) < self.config.MIN_SYNC_FRAMES:
            sample = "\n".join([f"{k}: {v}" for k, v in list(sync_groups.items())[:3]])
            raise ValueError(
                f"需要至少 {self.config.MIN_SYNC_FRAMES} 组同步帧，找到 {len(valid_groups)} 组\n"
                f"前3个时间戳示例:\n{sample}\n"
                "请检查: 1) 时间戳是否一致 2) 三个相机是否都有对应图像"
            )
        return valid_groups

    def _find_corners(self, image):
        """查找棋盘格角点（带错误处理）"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, self.config.PATTERN_SIZE,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return ret, corners
        except Exception as e:
            print(f"角点检测错误: {str(e)}")
            return False, None

    def _safe_imshow(self, win_name, image, delay=1):
        """安全的图像显示方法"""
        try:
            cv2.imshow(win_name, image)
            cv2.waitKey(delay)
            return True
        except Exception as e:
            print(f"显示错误: {str(e)}")
            return False

    def _draw_coordinate_axis(self, image, corners, K, dist, rvec, tvec):
        """绘制3D坐标系（带错误处理）"""
        try:
            axis = np.float32([
                [self.config.COORD_SCALE, 0, 0],
                [0, self.config.COORD_SCALE, 0],
                [0, 0, -self.config.COORD_SCALE]
            ]) * self.config.SQUARE_SIZE

            img_points, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
            origin = tuple(corners[0].ravel().astype(int))

            cv2.line(image, origin, tuple(img_points[0].ravel().astype(int)), (0, 0, 255), 3)
            cv2.line(image, origin, tuple(img_points[1].ravel().astype(int)), (0, 255, 0), 3)
            cv2.line(image, origin, tuple(img_points[2].ravel().astype(int)), (255, 0, 0), 3)
            return image
        except Exception as e:
            print(f"坐标系绘制失败: {str(e)}")
            return image

    def calibrate_single_camera(self, cam_id):
        """单相机标定（修复窗口错误）"""
        cam_idx = cam_id[-1]
        img_dir = Path(self.config.SINGLE_CALIB_DIR) / f"cam{cam_idx}"

        # 验证目录存在性
        if not img_dir.exists():
            raise FileNotFoundError(f"单目标定目录不存在: {img_dir}")
        if not list(img_dir.glob("*.png")):
            raise ValueError(f"目录中未找到PNG图像: {img_dir}")

        obj_points, img_points = [], []
        valid_images = []
        win_name = f"Calib-{cam_id}"
        window_created = False

        try:
            # 初始化窗口
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(win_name, 1280, 720)

            # 显示初始化画面
            placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"正在加载 {cam_id} 图像...",
                        (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self._safe_imshow(win_name, placeholder, 50)

            # 处理图像
            for img_path in img_dir.glob("*.png"):
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"无法读取图像: {img_path}")
                    continue

                preview = image.copy()
                ret, corners = self._find_corners(image)

                if ret:
                    obj_points.append(self.objp)
                    img_points.append(corners)
                    valid_images.append(str(img_path))

                    # 绘制检测结果
                    cv2.drawChessboardCorners(preview, self.config.PATTERN_SIZE, corners, ret)
                    cv2.putText(preview, "VALID", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(preview, "INVALID", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                self._safe_imshow(win_name, preview, 100)

            # 验证有效图像
            if len(valid_images) < self.config.MIN_VALID_IMAGES:
                raise ValueError(f"{cam_id} 有效图像不足 ({len(valid_images)}/{self.config.MIN_VALID_IMAGES})")

            # 执行标定
            h, w = image.shape[:2]
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, (w, h), None, None
            )

            # 保存结果
            self.calib_data[cam_id] = {
                "K": K.tolist(),
                "dist": dist.tolist(),
                "image_size": [w, h],
                "reproj_error": float(ret),
                "valid_images": valid_images,
                "rvecs": [rvec.tolist() for rvec in rvecs],
                "tvecs": [tvec.tolist() for tvec in tvecs]
            }

            # 显示结果
            self._show_calibration_result(cam_id)

        except Exception as e:
            print(f"标定失败: {str(e)}")
            print(traceback.format_exc())
            raise
        finally:
            # 安全关闭窗口
            if window_created:
                try:
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyWindow(win_name)
                        cv2.waitKey(1)
                except:
                    pass

    def _show_calibration_result(self, cam_id):
        """显示标定结果（带错误处理）"""
        try:
            cam_data = self.calib_data[cam_id]
            K = np.array(cam_data["K"])
            dist = np.array(cam_data["dist"])
            rvecs = [np.array(rvec) for rvec in cam_data["rvecs"]]
            tvecs = [np.array(tvec) for tvec in cam_data["tvecs"]]

            win_name = f"Result-{cam_id}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 1280, 720)

            for i, img_path in enumerate(cam_data["valid_images"]):
                image = cv2.imread(img_path)
                if image is None:
                    continue

                ret, corners = self._find_corners(image)
                if ret:
                    try:
                        image = self._draw_coordinate_axis(image, corners, K, dist, rvecs[i], tvecs[i])
                        cv2.putText(image, f"Error: {cam_data['reproj_error']:.2f}px",
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        self._safe_imshow(win_name, image, 500)
                    except Exception as e:
                        print(f"结果显示错误: {str(e)}")

            cv2.destroyAllWindows()
        except Exception as e:
            print(f"结果展示失败: {str(e)}")

    def calibrate_stereo(self, cam1, cam2):
        """立体标定（修复同步问题）"""
        obj_points, img_points1, img_points2 = [], [], []

        # 加载同步图像对
        for sync_group in self.sync_groups:
            img_path1 = sync_group.get(cam1)
            img_path2 = sync_group.get(cam2)
            if not (img_path1 and img_path2):
                continue

            # 读取并检测角点
            img1 = cv2.imread(img_path1)
            ret1, corners1 = self._find_corners(img1)

            img2 = cv2.imread(img_path2)
            ret2, corners2 = self._find_corners(img2)

            if ret1 and ret2:
                obj_points.append(self.objp)
                img_points1.append(corners1)
                img_points2.append(corners2)

        # 验证有效对数
        if len(obj_points) < 5:
            raise ValueError(f"有效同步对数不足 ({len(obj_points)}/5)")

        # 获取相机参数
        K1 = np.array(self.calib_data[cam1]['K'])
        dist1 = np.array(self.calib_data[cam1]['dist'])
        K2 = np.array(self.calib_data[cam2]['K'])
        dist2 = np.array(self.calib_data[cam2]['dist'])
        image_size = tuple(self.calib_data[cam1]['image_size'])

        # 执行立体标定
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points1, img_points2,
            K1, dist1, K2, dist2, image_size,
            flags=flags
        )

        return {
            "R": R.tolist(),
            "T": T.flatten().tolist(),
            "reproj_error": ret,
            "num_pairs": len(obj_points)
        }

    def build_global_extrinsics(self):
        """构建全局坐标系（以camera0为原点）"""
        extrinsics = {
            "camera0": {
                "R_world_to_cam": np.eye(3).tolist(),
                "T_world_to_cam": [0.0, 0.0, 0.0]
            }
        }

        # 计算其他相机相对于camera0的外参
        for cam_id in self.config.camera_paths:
            if cam_id == "camera0":
                continue

            try:
                stereo_data = self.calibrate_stereo("camera0", cam_id)
                print(
                    f"{cam_id} 标定结果: 误差={stereo_data['reproj_error']:.2f}px, 使用帧数={stereo_data['num_pairs']}")

                extrinsics[cam_id] = {
                    "R_world_to_cam": stereo_data["R"],
                    "T_world_to_cam": stereo_data["T"]
                }
            except Exception as e:
                print(f"{cam_id} 外参计算失败: {str(e)}")
                raise

        return extrinsics


# ================== 主程序 ==================
if __name__ == "__main__":
    cfg = CheckerboardConfig()
    calibrator = None

    try:
        # 创建目录
        os.makedirs(cfg.SINGLE_CALIB_DIR, exist_ok=True)
        os.makedirs(cfg.SYNC_CALIB_DIR, exist_ok=True)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # 初始化标定器
        calibrator = CheckerboardCalibrator(cfg)
        print(f"找到 {len(calibrator.sync_groups)} 组有效同步帧")

        print("\n步骤1/3: 单相机内参标定")
        for cam_id in cfg.camera_paths:
            print(f"正在处理 {cam_id}...", end="", flush=True)
            calibrator.calibrate_single_camera(cam_id)
            error = calibrator.calib_data[cam_id]["reproj_error"]
            print(f"完成！重投影误差: {error:.2f}px")

        print("\n步骤2/3: 多相机外参标定")
        extrinsics = calibrator.build_global_extrinsics()

        print("\n步骤3/3: 保存结果")
        output_data = {
            "intrinsics": calibrator.calib_data,
            "extrinsics": extrinsics,
            "sync_groups": len(calibrator.sync_groups),
            "config": {
                "pattern_size": cfg.PATTERN_SIZE,
                "square_size": cfg.SQUARE_SIZE
            }
        }

        output_path = Path(cfg.OUTPUT_DIR) / "calibration.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"标定完成！结果已保存至:\n{output_path.resolve()}")

    except Exception as e:
        print(f"\n标定失败: {str(e)}")
        if calibrator:
            print(f"有效同步组数: {len(calibrator.sync_groups)}")
        print(traceback.format_exc())
    finally:
        cv2.destroyAllWindows()