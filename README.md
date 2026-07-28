markdown
# 平安城市智能巡检机器人仿真系统

> 基于 ROS 与 Gazebo 的自主巡检机器人仿真项目，面向平安城市/全域安防场景，实现环境感知、地图构建、路径导航、目标识别与结果输出等完整技术链路。

---

## 📌 项目简介

本项目针对平安城市移动机器人在仿真城市场景中的自主巡检任务需求，设计并实现了一套基于 **ROS（Noetic）** 与 **Gazebo** 的智能巡检机器人仿真系统。

系统在 Gazebo 中构建了包含楼宇、垃圾桶、人群和道路边界等元素的 **4000 mm × 4000 mm** 仿真场地，并以移动机器人平台为载体，集成激光雷达、RealSense D435 相机和里程计等传感器，为环境感知、建图定位和目标识别提供数据基础。

**完整技术链路**：
仿真场景构建 → 传感器感知 → 地图生成与定位 → 导航避障 → 多点巡检 → 目标识别 → 结果反馈

text

---

## 🚀 快速开始

### 环境依赖

- **操作系统**：Ubuntu 20.04
- **ROS 发行版**：Noetic
- **Gazebo**：9.0+
- **Python**：3.8+

### 安装与编译

```bash
# 1. 克隆项目
cd ~/catkin_ws/src
git clone <your-repo-url>

# 2. 安装依赖
rosdep install --from-paths . --ignore-src -r -y

# 3. 编译
cd ~/catkin_ws
catkin_make
source devel/setup.bash
启动仿真
bash
# 启动仿真世界 + 机器人
roslaunch saft_city_gazebo saft_city_world.launch

# 启动导航与 SLAM（另开终端）
roslaunch saft_city_navigation navigation.launch

# 启动视觉识别节点（另开终端）
roslaunch saft_city_perception detection.launch

# 启动巡航任务（另开终端）
rosrun saft_city_perception patrol_route.py
交互操作
空格键：在识别节点运行状态下，按空格保存当前画面到本地（用于结果记录）。

路径点发布：通过 waypoint_nav.py 脚本可动态发布巡检目标点。

🗺️ 系统架构与核心功能
模块	功能描述
仿真环境	Gazebo 构建 4m×4m 城市场景，含楼宇、人群、垃圾桶、道路边界
机器人平台	Clearpath Husky + RealSense D435 + SICK LMS1XX 激光雷达
SLAM 建图	gmapping + 本队独立开发的 Gazebo 坐标解析脚本（生成 PGM 栅格地图）
导航规划	move_base + global_planner + DWA + robot_localization（EKF 融合）
目标识别	PP-YOLOE+ S（10 类安防目标：垃圾桶、人群、楼宇异常等）
任务调度	12 点混合巡航（路过点预切 + 停靠点精确到位 + 开环转向）
🔧 核心参数说明
传感器安装（Husky 定制）
bash
# RealSense D435 安装位置（前移 12cm，抬高 28cm，俯仰 8.6°）
HUSKY_REALSENSE_XYZ="0.12 0 0.28"
HUSKY_REALSENSE_RPY="0 0.15 0"
HUSKY_SENSOR_ARCH=1
HUSKY_LMS1XX_ENABLED=1
DWA 局部规划器（关键参数）
参数	取值	说明
max_vel_x	0.6 m/s	最大前进速度
occdist_scale	0.2	障碍物距离代价权重
path_distance_bias	40.0	路径跟随权重
sim_time	2.0 s	轨迹模拟时长
代价地图
参数	取值
全局地图尺寸	6m × 6m（滚动窗口）
分辨率	0.01 m/px
膨胀半径	0.06 m
📁 项目文件结构
text
saft_city_ws/
├── src/
│   ├── saft_city_gazebo/
│   │   ├── launch/
│   │   │   ├── saft_city_world.launch
│   │   │   └── navigation.launch
│   │   └── worlds/
│   │       └── circular_track_walls.world
│   ├── saft_city_navigation/
│   │   ├── launch/
│   │   │   ├── move_base.launch
│   │   │   └── gmapping_slam.launch
│   │   └── config/
│   │       ├── costmap_common_params.yaml
│   │       ├── costmap_global_params.yaml
│   │       ├── costmap_local_params.yaml
│   │       ├── dwa_local_planner_params.yaml
│   │       └── global_planner_params.yaml
│   └── saft_city_perception/
│       ├── scripts/
│       │   ├── patrol_route.py
│       │   ├── waypoint_nav.py
│       │   ├── inference_node.py
│       │   └── photo_capture.py
│       ├── inference/
│       │   └── infer.py
│       └── training/
│           ├── train.py
│           └── augment.py
📜 开源许可证与第三方依赖声明
本项目基于 ROS 和 Gazebo 仿真平台开发，使用了以下开源组件，并严格遵守其许可证条款：

模块分类	开源项目名称	许可证	本团队原创/修改说明
机器人模型	Clearpath Husky	BSD 3-Clause	加装 RealSense D435 与传感器支架；自定义传感器安装位置及俯仰角参数
传感器驱动	Intel RealSense D435 / SICK LMS1XX	BSD / 专有 SDK	相机前移 12cm、抬高 28cm、俯仰 8.6°；激光雷达发布 /front/scan 话题
SLAM 建图	gmapping（ROS）	BSD 3-Clause	本队独立开发 Gazebo 坐标解析脚本，渲染 PGM 栅格地图（分辨率 0.01m/px）
导航框架	move_base / global_planner / DWA	BSD 3-Clause	DWA 参数调优（path_distance_bias=40.0 等）；12 点混合巡航模式
位姿融合	robot_localization	BSD 3-Clause	EKF 融合轮速里程计 + IMU
目标检测	PP-YOLOE+ S（PaddleDetection）	Apache 2.0	基于预训练权重训练 10 类安防目标模型；编写 TCP 推流与结果回传节点
图像处理	OpenCV	BSD 3-Clause	图像预处理与标注结果可视化
本项目的原创工作包括：

独立开发的 Gazebo 模型坐标解析与 PGM 栅格地图生成脚本（map_generator.py）；

基于 ROS Action 的 12 点混合巡航任务调度逻辑（patrol_route.py）；

RealSense 图像 TCP 推流与检测结果回传机制（stream_server.py / inference_node.py）；

仿真场地的完整建模（楼宇、人群、垃圾桶、道路边界等）。

📌 第三方依赖详细声明
本项目使用的第三方开源组件详细信息如下：

1. ROS (Robot Operating System)

许可证：BSD 3-Clause License

使用方式：核心通信框架与工具链基础

版权声明：Copyright (c) 2008-2023, Open Source Robotics Foundation

2. Gazebo 仿真平台

许可证：Apache 2.0 License

使用方式：构建仿真场地与机器人物理引擎

版权声明：Copyright (c) 2012-2023, Open Source Robotics Foundation

3. Clearpath Husky 机器人模型

许可证：BSD 3-Clause License

使用方式：移动机器人基础平台

版权声明：Copyright (c) 2014, Clearpath Robotics

4. PaddleDetection / PP-YOLOE+

许可证：Apache 2.0 License

使用方式：目标检测模型训练框架

版权声明：Copyright (c) 2020-2023, PaddlePaddle Authors

特别说明：本仓库中如包含 PP-YOLOE+ 预训练权重文件（.pdparams），均来自 PaddleDetection 官方仓库，遵循 Apache 2.0 许可证。

5. OpenCV

许可证：BSD 3-Clause License

使用方式：图像预处理、标注结果可视化

版权声明：Copyright (c) 2000-2023, Intel Corporation, Willow Garage, etc.

6. robot_localization

许可证：BSD 3-Clause License

使用方式：EKF 融合轮速里程计与 IMU 位姿

版权声明：Copyright (c) 2012-2023, Charles River Analytics

7. Navigation Stack (move_base / DWA / global_planner)

许可证：BSD 3-Clause License

使用方式：全局与局部路径规划框架

版权声明：Copyright (c) 2008-2023, Willow Garage, Inc. / Open Source Robotics Foundation

以上各许可证的完整文本可通过 https://opensource.org/licenses/BSD-3-Clause 和 https://www.apache.org/licenses/LICENSE-2.0 查阅。

👥 团队与致谢
本项目为平安城市/全域安防巡检竞赛参赛作品。

感谢以下开源社区提供的优秀工具与框架：

ROS (BSD)

Gazebo (Apache 2.0)

Clearpath Husky (BSD)

PaddleDetection (Apache 2.0)

OpenCV (BSD)
最后更新：2026 年 7 月
