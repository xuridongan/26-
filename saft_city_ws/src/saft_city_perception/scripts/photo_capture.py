#!/usr/bin/env python3
"""拍照工具 — 按空格保存相机画面到桌面"""
import rospy, cv2, os, threading, sys, select, termios, tty
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class PhotoCapture:
    def __init__(self):
        rospy.init_node('photo_capture')
        self.bridge = CvBridge()
        self.latest = None
        self.save_dir = os.path.expanduser('~/桌面/model')
        os.makedirs(self.save_dir, exist_ok=True)
        self.count = len([f for f in os.listdir(self.save_dir) if f.startswith('photo_')]) + 1

        rospy.Subscriber('/realsense/color/image_raw', Image, self.img_cb)
        rospy.loginfo(f"✅ 拍照工具就绪！按空格拍照，保存到 {self.save_dir}")

    def img_cb(self, msg):
        self.latest = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def run(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == ' ':
                        self.take_photo()
                    elif key == 'q':
                        break
                rospy.sleep(0.05)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def take_photo(self):
        if self.latest is None:
            print("⚠️ 暂无画面")
            return
        path = f"{self.save_dir}/photo_{self.count:03d}.jpg"
        cv2.imwrite(path, self.latest)
        print(f"📸 {path}")
        self.count += 1

if __name__ == '__main__':
    c = PhotoCapture()
    c.run()
