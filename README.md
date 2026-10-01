# Three-Camera Mouse 3D Reconstruction

三相机小鼠动作捕捉与三维重建工程的公开 Demo。项目面向转台上的小鼠行为实验：同步采集三路视频，使用棋盘格完成相机标定，再将 DeepLabCut 输出的 2D 关键点重建为三维轨迹，为术前/术后小鼠抗眩晕能力比较提供运动学基础。

## 项目链路

```text
三相机采集 → 三相机内外参标定 → 2D 关键点质量筛选
          → 三视角 RANSAC 三角测量 → 轨迹插值/速度约束滤波
          → 三维骨架回投影 → 术前/术后运动学指标比较
```

## 公开代码

- `capture_three_cameras.py`：三路 Orbbec 相机线程采集、预览、录像和同步快照。
- `calibrate_three_cameras.py`：11×8 棋盘格单目标定、三相机外参标定和全局坐标系构建。
- `mouse_triangulation.py`：从三路 2D 关键点和投影矩阵进行置信度筛选、RANSAC 三角测量与缺失轨迹插值。
- `trajectory_metrics.py`：计算三维质心路径、速度、位移和有效轨迹比例，支持术前/术后条件比较。
- `research_reconstruct_3d.py`：从实际工程抽取的三只小鼠批量重建与二维/三维轨迹滤波流程，保留原始研究代码结构供复核。
- `demo_synthetic.py`：使用合成相机和小鼠骨架验证三视角重建流程。

## 运行公开 Demo

```powershell
python -m pip install -r requirements.txt
python demo_synthetic.py
python -m pytest -q
```

Demo 不需要真实动物数据、视频或相机设备。运行结果会报告合成三维关键点重建误差和轨迹指标。

## 实际设备采集

真实采集还需要安装 Orbbec SDK 的 Python binding，并根据设备型号调整 `capture_three_cameras.py` 中的流配置。标定图像、视频、DeepLabCut 输出、标定 JSON 和实验日志不包含在仓库中。

## 简历描述

> 面向转台小鼠行为实验搭建三相机三维动作捕捉流水线，完成多相机同步采集、棋盘格内外参标定、DeepLabCut 关键点三视角 RANSAC 重建、轨迹滤波及术前/术后运动学指标提取。

## 研究边界

本仓库展示三维重建和运动学工程能力，不包含真实动物实验结果，也不对术前/术后抗眩晕差异作结论。
