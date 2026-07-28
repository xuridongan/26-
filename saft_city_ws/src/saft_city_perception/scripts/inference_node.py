#!/usr/bin/env python3
"""
实时推理节点: 订阅相机图像 → Paddle模型推理 → 绘制结果 → 发布标注画面

启动前先:
  pip3 install paddlepaddle paddledet  (或 paddlepaddle-gpu 如有GPU)
"""
import rospy, cv2, os, json, sys
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'inference')

class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node')
        self.bridge = CvBridge()
        self.model = None
        self.model_loaded = False

        # 参数
        self.score_thresh = rospy.get_param('~score_thresh', 0.25)
        self.proc_rate = rospy.get_param('~proc_rate', 3)  # CPU推理约3fps
        self.input_topic = rospy.get_param('~input_topic', '/realsense/color/image_raw')

        # 类别阈值（JSON中key是字符串"1"~"10"，转成0索引列表）
        self.class_thresholds = None
        thresh_path = os.path.join(MODEL_DIR, 'class_thresholds.json')
        if os.path.exists(thresh_path):
            with open(thresh_path) as f:
                raw = json.load(f)
            self.class_thresholds = [raw.get(str(i+1), 0.05) for i in range(10)]
            rospy.loginfo(f"Loaded class thresholds, {len(self.class_thresholds)} classes")

        # 类别名称与颜色
        self.class_names = [
            "collapsed building", "building on fire", "building with toxic gas",
            "building with a power failure", "medical rescue groups", "general rescue groups",
            "kitchen waste bin", "recyclable waste bin", "hazardous waste bin", "residual waste bin",
        ]
        self.colors = [
            (0,0,255), (255,0,0), (0,255,255), (255,0,255),
            (0,255,0), (255,255,0), (128,0,128), (255,165,0), (128,128,0), (0,128,128),
        ]

        # 加载模型
        try:
            self._load_model()
        except Exception as e:
            rospy.logwarn(f"模型加载失败: {e}，推理功能不可用")
            rospy.logwarn("请安装: pip3 install paddlepaddle paddledet")

        # 订阅相机
        self.image_sub = rospy.Subscriber(self.input_topic, Image, self.image_cb, queue_size=1)

        # 发布标注画面
        self.result_pub = rospy.Publisher('/detection/result_image', Image, queue_size=10)
        self.text_pub = rospy.Publisher('/detection/result_text', String, queue_size=10)

        self.last_img = None
        self.timer = rospy.Timer(rospy.Duration(1.0/self.proc_rate), self.process)

        rospy.loginfo(f"Inference node started, input={self.input_topic}, rate={self.proc_rate}hz")

    def _load_model(self):
        sys.path.insert(0, MODEL_DIR)
        from infer import build_infer_model
        weights = os.path.join(MODEL_DIR, 'model_final.pdparams')
        if not os.path.exists(weights):
            raise FileNotFoundError(f"模型权重不存在: {weights}")
        self.model = build_infer_model(0.01, weights)
        self.model_loaded = True
        rospy.loginfo("模型加载成功!")

    def image_cb(self, msg):
        self.last_img = msg

    def process(self, event):
        if self.last_img is None or not self.model_loaded:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.last_img, "bgr8")
            result_img, detections = self._infer_and_draw(cv_img)

            # 发布标注画面
            self.result_pub.publish(self.bridge.cv2_to_imgmsg(result_img, "bgr8"))

            # 发布文本结果
            if detections:
                text = "; ".join([f"{d['class_name']}({d['score']:.2f})" for d in detections[:5]])
                self.text_pub.publish(String(text))

        except Exception as e:
            rospy.logwarn_throttle(5.0, f"推理异常: {e}")

    def _infer_and_draw(self, cv_img):
        """推理并绘制边框"""
        from infer import infer_image_array
        dets = infer_image_array(self.model, cv_img, self.score_thresh, self.class_thresholds)

        for d in dets:
            x1, y1, x2, y2 = d["bbox_pixel"]
            cls_id = d["class_id"]
            score = d["score"]
            label = f"{d['class_name']} {score:.2f}"

            color = self.colors[cls_id % len(self.colors)]
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 叠加统计信息
        if dets:
            text = f"Detected: {len(dets)} objects"
            cv2.putText(cv_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # 分类统计
            from collections import Counter
            counts = Counter(d["class_name"] for d in dets)
            y = 60
            for name, cnt in counts.most_common(5):
                cv2.putText(cv_img, f"  {name}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                y += 25

        return cv_img, dets

if __name__ == '__main__':
    try:
        InferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
