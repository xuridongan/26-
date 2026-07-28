#!/usr/bin/env python3
"""
saft_city_perception - Object Detection Integration Node

集成接口节点：接收您训练的模型输出，将识别结果通过 ROS 话题发布，
并在终端打印（ROS_INFO）以便满足竞赛"关键输出到终端"的要求。

待集成说明：
  1. 将您训练的模型（如 ONNX / TensorRT / PyTorch）加载到 model 变量中
  2. 在 process_image() 中调用模型推理，解析结果
  3. 识别结果会通过 /detection/objects 话题发布，同时打印到终端
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import String


class ModelIntegrationNode:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=False)

        self.bridge = CvBridge()
        self.latest_image = None
        self.processing_rate = rospy.get_param('~processing_rate', 5)

        # ====================================================
        # 在这里加载您的模型，例如：
        # self.model = load_your_model("path/to/model.onnx")
        # ====================================================
        self.model = None
        rospy.loginfo("[Model] 请在此加载您的训练模型 (ONNX/TensorRT/PyTorch)")

        # 发布检测结果
        self.detection_pub = rospy.Publisher('/detection/objects', String, queue_size=10)

        # 订阅相机图像
        self.image_sub = rospy.Subscriber(
            '/realsense/color/image_raw', Image, self.image_callback, queue_size=1)

        # 定时处理
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.processing_rate), self.process)

        rospy.loginfo("Object Detection Integration Node 已启动")
        rospy.loginfo("请在 object_detector.py 中加载您的模型并实现推理逻辑")

    def image_callback(self, msg):
        self.latest_image = msg

    def process(self, event):
        if self.latest_image is None:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")

            # ====================================================
            # 在这里调用您的模型推理，例如：
            # results = self.model.infer(cv_img)
            #
            # 从 results 中解析出:
            #   - 垃圾桶类型 (recyclable/kitchen/hazard)
            #   - 人群数量与类型
            #   - 楼宇灾害情况
            # ====================================================

            # ---------- 占位输出示例 (请替换为真实推理结果) ----------
            trash_bins = []       # [{"type": "recycle"}, ...]
            people_count = 0
            building_status = []  # [{"id": 1, "status": "fire"}, ...]

            if self.model is not None:
                # TODO: 实际推理
                pass

            # 终端打印 (满足竞赛要求)
            print("\n========== 平安城市 - 识别结果 ==========")
            print(f"【垃圾桶】{trash_bins if trash_bins else '未检测到'}")
            print(f"【人  群】{people_count} 人")
            print(f"【楼  宇】{building_status if building_status else '未检测到'}")
            print("============================================\n")

            # 发布结果话题
            msg = String()
            msg.data = (
                f"trash_bins={trash_bins}|people={people_count}|"
                f"buildings={building_status}"
            )
            self.detection_pub.publish(msg)
            # ====================================================

        except Exception as e:
            rospy.logwarn("处理异常: %s", str(e))


if __name__ == '__main__':
    try:
        node = ModelIntegrationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
