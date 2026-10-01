import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn.utils import check_random_state
from scipy.interpolate import interp1d

# ==================== 配置参数 ====================
DLC_CSV_PATHS = [
    "2d_csv/mouse/camera0_keypoints.csv",
    "2d_csv/mouse/camera1_keypoints.csv",
    "2d_csv/mouse/camera2_keypoints.csv",
]
CALIB_FILE = "calibration_results/calibration.json"
OUTPUT_NPY_RAW = "3d_point/3d_mice_points_raw.npy"
OUTPUT_CSV_RAW = "3d_point/3d_mice_points_raw.csv"
OUTPUT_NPY_FILTERED = "3d_point/3d_mice_points_filtered.npy"
OUTPUT_CSV_FILTERED = "3d_point/3d_mice_points_filtered.csv"
N_MICE = 3
BODYPARTS = ["mouth", "L_ear", "R_ear", "tail",
             "L_Ffoot", "R_Ffoot", "L_Bfoot", "R_Bfoot"]
N_JOINTS = len(BODYPARTS)
INITIAL_BAD_FRAMES = 10
CONF_THRESHOLD = 0.8
REPROJ_THRESHOLD = 15.0

# 滤波参数
FILTER_WINDOW = 5  # 滑动窗口大小
MAX_SPEED = 50.0   # 最大允许速度（像素/帧）
MAX_ACCEL = 20.0   # 最大允许加速度（像素/帧²）
MAX_3D_SPEED = 0.03  # 最大允许3D速度（米/帧）
MAX_3D_ACCEL = 0.015  # 最大允许3D加速度（米/帧²）


# ==================== 标定数据加载 ====================
class CalibrationValidator:
    def __init__(self, calib_file):
        with open(calib_file) as f:
            self.calib = json.load(f)
        self.cam_params = self._load_params()

    def _load_params(self):
        cam_params = {}
        for cam_name in ["camera0", "camera1", "camera2"]:
            K = np.array(self.calib["intrinsics"][cam_name]["K"], dtype=np.float32)
            R = np.array(self.calib["extrinsics"][cam_name]["R_world_to_cam"], dtype=np.float32)
            T = np.array(self.calib["extrinsics"][cam_name]["T_world_to_cam"], dtype=np.float32)
            RT = np.hstack([R, T.reshape(-1, 1)])

            cam_params[cam_name] = {
                "P": K @ RT,
                "K": K,
                "R": R,
                "T": T,
                "dist": np.array(self.calib["intrinsics"][cam_name]["dist"][0]),
                "image_size": tuple(self.calib["intrinsics"][cam_name]["image_size"])
            }
        return cam_params

    def check_extrinsics(self):
        print("\n外参验证:")
        for cam in ["camera1", "camera2"]:
            T = self.cam_params[cam]["T"]
            print(f"{cam} 距基准相机的距离: {np.linalg.norm(T):.2f}m")


# ==================== 数据加载器 ====================
class MultiMouseLoader:
    def __init__(self, csv_paths):
        self.csv_paths = csv_paths
        self.individuals = [f"individual{i + 1}" for i in range(N_MICE)]

    def _validate_structure(self, df):
        """验证CSV文件结构"""
        try:
            actual_bps = df.columns.get_level_values(2).unique().tolist()
            assert actual_bps == BODYPARTS, "关节点不匹配"
            assert df.columns.nlevels == 4, "表头层级错误"
        except Exception as e:
            raise ValueError(f"CSV结构验证失败: {str(e)}")

    def load(self):
        mice_data = []
        frame_counts = []

        # 第一遍：验证结构和帧数
        for path in self.csv_paths:
            df = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0, low_memory=False)
            self._validate_structure(df)
            frame_counts.append(len(df))

        if len(set(frame_counts)) != 1:
            raise ValueError(f"帧数不一致: {frame_counts}")
        n_frames = frame_counts[0]

        # 第二遍：加载数据
        for mouse_idx in range(N_MICE):
            mouse_cams = []
            for cam_idx, path in enumerate(self.csv_paths):
                df = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0, low_memory=False)
                cam_data = np.full((n_frames, N_JOINTS, 3), np.nan)

                for joint_idx, bp in enumerate(BODYPARTS):
                    try:
                        individual = self.individuals[mouse_idx]
                        x = df.xs((individual, bp, 'x'), axis=1, level=[1, 2, 3]).values.ravel()
                        y = df.xs((individual, bp, 'y'), axis=1, level=[1, 2, 3]).values.ravel()
                        conf = df.xs((individual, bp, 'likelihood'), axis=1, level=[1, 2, 3]).values.ravel()

                        valid = (conf >= CONF_THRESHOLD) & ~np.isnan(x) & ~np.isnan(y)
                        cam_data[valid, joint_idx, 0] = x[valid]
                        cam_data[valid, joint_idx, 1] = y[valid]
                        cam_data[valid, joint_idx, 2] = conf[valid]
                    except KeyError:
                        continue

                # 过滤初始帧
                cam_data[:INITIAL_BAD_FRAMES] = np.nan
                
                # 应用坐标滤波
                cam_data = CoordinateFilter.filter_coordinates(cam_data)
                
                mouse_cams.append(cam_data)

            mice_data.append(np.array(mouse_cams))

        return np.array(mice_data)  # (mice, cams, frames, joints, coords)


# ==================== 三维重建器 ====================
class MultiMouseTriangulator:
    def __init__(self, cam_params):
        self.cam_params = cam_params

    def triangulate_mice(self, all_mice_2d):
        """
        输入: all_mice_2d 形状 (mice, cams, frames, joints, coords)
        输出: points_3d 形状 (frames, mice, joints, 3)
        """
        n_mice, n_cams, n_frames, n_joints, _ = all_mice_2d.shape
        points_3d = np.full((n_frames, n_mice, n_joints, 3), np.nan)

        for mouse_idx in range(n_mice):
            print(f"\nProcessing Mouse {mouse_idx + 1}")
            for joint in tqdm(range(n_joints), desc="Joints", unit="joint"):
                for frame in range(n_frames):
                    observations = [
                        (
                            all_mice_2d[mouse_idx, cam_idx, frame, joint, 0],
                            all_mice_2d[mouse_idx, cam_idx, frame, joint, 1],
                            all_mice_2d[mouse_idx, cam_idx, frame, joint, 2]
                        )
                        for cam_idx in range(n_cams)
                    ]
                    points_3d[frame, mouse_idx, joint] = self._ransac_triangulate(observations)
        return points_3d

    def _ransac_triangulate(self, observations, max_iter=100):
        valid_views = [i for i, (x, y, c) in enumerate(observations)
                       if (c >= CONF_THRESHOLD) and not np.isnan(x) and not np.isnan(y)]

        if len(valid_views) < 2:
            return np.full(3, np.nan)

        best_inliers = []
        best_point = None
        rng = check_random_state(42)

        for _ in range(max_iter):
            sample = rng.choice(valid_views, size=min(3, len(valid_views)), replace=False)
            sample_pts = [(observations[i][0], observations[i][1]) for i in sample]

            try:
                point = self._triangulate(sample_pts, sample)
            except np.linalg.LinAlgError:
                continue

            inliers = []
            for i in valid_views:
                error = self._reprojection_error(point, observations[i][:2], i)
                if error < REPROJ_THRESHOLD:
                    inliers.append(i)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_point = point

        return best_point if best_point is not None else np.full(3, np.nan)

    def _triangulate(self, points, cam_indices):
        A = []
        for (x, y), idx in zip(points, cam_indices):
            P = self.cam_params[f"camera{idx}"]["P"]
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])

        _, _, Vt = np.linalg.svd(np.array(A))
        point = Vt[-1, :4]
        # 直接将毫米数值作为米数值
        return (point[:3] / point[3]).astype(np.float32)

    def _reprojection_error(self, point3d, point2d, cam_idx):
        if np.isnan(point3d).any():
            return float('inf')
        P = self.cam_params[f"camera{cam_idx}"]["P"]
        # 直接将米数值作为毫米数值进行重投影
        proj = P @ np.append(point3d, 1.0)
        proj = proj[:2] / proj[2]
        return np.linalg.norm(point2d - proj)


# ==================== 后处理器 ====================
class PostProcessor:
    @staticmethod
    def process(points_3d):
        """轨迹插值与平滑"""
        n_frames, n_mice, n_joints, _ = points_3d.shape
        interpolated = np.copy(points_3d)

        for m in range(n_mice):
            for j in range(n_joints):
                valid = ~np.isnan(points_3d[:, m, j, 0])
                if valid.sum() < 2:
                    continue

                x = interp1d(np.where(valid)[0], points_3d[valid, m, j],
                             axis=0, kind='linear', fill_value="extrapolate")
                interpolated[:, m, j] = x(np.arange(n_frames))

        return interpolated


# ==================== 二维坐标滤波 ====================
class CoordinateFilter:
    @staticmethod
    def filter_coordinates(coordinates, window_size=FILTER_WINDOW, max_speed=MAX_SPEED, max_accel=MAX_ACCEL):
        """
        对二维坐标进行滤波，去除突变点
        coordinates: shape (frames, joints, 3) - 包含x,y,conf的坐标数组
        """
        filtered = np.copy(coordinates)
        n_frames, n_joints, _ = coordinates.shape
        
        for joint in range(n_joints):
            # 获取有效点的索引
            valid_mask = ~np.isnan(coordinates[:, joint, 0])
            if np.sum(valid_mask) < 2:
                continue
                
            # 计算速度和加速度
            x = coordinates[:, joint, 0]
            y = coordinates[:, joint, 1]
            
            # 使用中值滤波去除突变
            x_filtered = np.copy(x)
            y_filtered = np.copy(y)
            
            for i in range(window_size, n_frames - window_size):
                if valid_mask[i]:
                    # 获取窗口内的有效点
                    window_x = x[i-window_size:i+window_size+1]
                    window_y = y[i-window_size:i+window_size+1]
                    window_valid = valid_mask[i-window_size:i+window_size+1]
                    
                    if np.sum(window_valid) >= 3:  # 至少需要3个点
                        # 计算中值
                        x_filtered[i] = np.median(window_x[window_valid])
                        y_filtered[i] = np.median(window_y[window_valid])
                        
                        # 计算速度和加速度
                        if i > 0 and valid_mask[i-1]:
                            dx = x_filtered[i] - x_filtered[i-1]
                            dy = y_filtered[i] - y_filtered[i-1]
                            speed = np.sqrt(dx*dx + dy*dy)
                            
                            if speed > max_speed:
                                # 如果速度过大，使用线性插值
                                x_filtered[i] = x_filtered[i-1] + dx * (max_speed/speed)
                                y_filtered[i] = y_filtered[i-1] + dy * (max_speed/speed)
            
            filtered[:, joint, 0] = x_filtered
            filtered[:, joint, 1] = y_filtered
            
        return filtered


# ==================== 三维坐标滤波 ====================
class ThreeDCoordinateFilter:
    @staticmethod
    def filter_coordinates(coordinates, window_size=FILTER_WINDOW, max_speed=MAX_3D_SPEED, max_accel=MAX_3D_ACCEL):
        """
        对三维坐标进行滤波，去除突变点
        coordinates: shape (frames, mice, joints, 3) - 包含x,y,z的坐标数组
        """
        filtered = np.copy(coordinates)
        n_frames, n_mice, n_joints, _ = coordinates.shape
        
        for mouse in range(n_mice):
            for joint in range(n_joints):
                # 获取有效点的索引
                valid_mask = ~np.isnan(coordinates[:, mouse, joint, 0])
                if np.sum(valid_mask) < 2:
                    continue
                
                # 获取坐标
                x = coordinates[:, mouse, joint, 0]
                y = coordinates[:, mouse, joint, 1]
                z = coordinates[:, mouse, joint, 2]
                
                # 使用中值滤波去除突变
                x_filtered = np.copy(x)
                y_filtered = np.copy(y)
                z_filtered = np.copy(z)
                
                for i in range(window_size, n_frames - window_size):
                    if valid_mask[i]:
                        # 获取窗口内的有效点
                        window_x = x[i-window_size:i+window_size+1]
                        window_y = y[i-window_size:i+window_size+1]
                        window_z = z[i-window_size:i+window_size+1]
                        window_valid = valid_mask[i-window_size:i+window_size+1]
                        
                        if np.sum(window_valid) >= 3:  # 至少需要3个点
                            # 计算中值
                            x_filtered[i] = np.median(window_x[window_valid])
                            y_filtered[i] = np.median(window_y[window_valid])
                            z_filtered[i] = np.median(window_z[window_valid])
                            
                            # 计算3D速度和加速度
                            if i > 0 and valid_mask[i-1]:
                                dx = x_filtered[i] - x_filtered[i-1]
                                dy = y_filtered[i] - y_filtered[i-1]
                                dz = z_filtered[i] - z_filtered[i-1]
                                speed = np.sqrt(dx*dx + dy*dy + dz*dz)
                                
                                if speed > max_speed:
                                    # 如果速度过大，使用线性插值
                                    x_filtered[i] = x_filtered[i-1] + dx * (max_speed/speed)
                                    y_filtered[i] = y_filtered[i-1] + dy * (max_speed/speed)
                                    z_filtered[i] = z_filtered[i-1] + dz * (max_speed/speed)
                
                filtered[:, mouse, joint, 0] = x_filtered
                filtered[:, mouse, joint, 1] = y_filtered
                filtered[:, mouse, joint, 2] = z_filtered
        
        return filtered


# ==================== 主流程 ====================
if __name__ == "__main__":
    # 初始化验证
    validator = CalibrationValidator(CALIB_FILE)
    validator.check_extrinsics()

    # 加载数据
    loader = MultiMouseLoader(DLC_CSV_PATHS)
    mice_2d = loader.load()
    print("数据加载完成，维度:", mice_2d.shape)

    # 三维重建
    triangulator = MultiMouseTriangulator(validator.cam_params)
    points_3d = triangulator.triangulate_mice(mice_2d)

    # 后处理
    points_3d = PostProcessor.process(points_3d)
    print("有效点比例:", np.mean(~np.isnan(points_3d)) * 100)

    # 保存原始NPY
    np.save(OUTPUT_NPY_RAW, points_3d)
    print(f"原始NPY文件已保存: {OUTPUT_NPY_RAW}")

    # 保存原始CSV
    csv_data = []
    for frame_idx in range(points_3d.shape[0]):
        frame_data = {"Frame": frame_idx}
        for m in range(N_MICE):
            for j, bp in enumerate(BODYPARTS):
                x, y, z = points_3d[frame_idx, m, j]
                frame_data.update({
                    f"Mouse{m}_{bp}_X": x,
                    f"Mouse{m}_{bp}_Y": y,
                    f"Mouse{m}_{bp}_Z": z
                })
        csv_data.append(frame_data)

    pd.DataFrame(csv_data).to_csv(OUTPUT_CSV_RAW, index=False)
    print(f"原始CSV文件已保存: {OUTPUT_CSV_RAW}")

    # 对三维坐标进行滤波
    points_3d_filtered = ThreeDCoordinateFilter.filter_coordinates(points_3d)
    print("三维坐标滤波完成")

    # 保存滤波后的NPY
    np.save(OUTPUT_NPY_FILTERED, points_3d_filtered)
    print(f"滤波后NPY文件已保存: {OUTPUT_NPY_FILTERED}")

    # 保存滤波后的CSV
    csv_data = []
    for frame_idx in range(points_3d_filtered.shape[0]):
        frame_data = {"Frame": frame_idx}
        for m in range(N_MICE):
            for j, bp in enumerate(BODYPARTS):
                x, y, z = points_3d_filtered[frame_idx, m, j]
                frame_data.update({
                    f"Mouse{m}_{bp}_X": x,
                    f"Mouse{m}_{bp}_Y": y,
                    f"Mouse{m}_{bp}_Z": z
                })
        csv_data.append(frame_data)

    pd.DataFrame(csv_data).to_csv(OUTPUT_CSV_FILTERED, index=False)
    print(f"滤波后CSV文件已保存: {OUTPUT_CSV_FILTERED}")
