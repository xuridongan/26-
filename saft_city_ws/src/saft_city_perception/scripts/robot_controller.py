#!/usr/bin/env python3
"""键盘控制: WASD移动 空格拍照 实时相机（Husky + RealSense）"""
import rospy, cv2, os, sys, threading, select
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller')
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=10)
        self.latest = None
        self.img_sub = rospy.Subscriber('/realsense/color/image_raw', Image, self.img_cb, queue_size=1)
        self.photo_dir = '/home/xuan/桌面/model'
        os.makedirs(self.photo_dir, exist_ok=True)
        self.photo_count = len([f for f in os.listdir(self.photo_dir) if f.startswith('photo_')]) + 1

    def img_cb(self, msg):
        self.latest = msg

    def show_camera(self):
        cv2.namedWindow('相机', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('相机', 640, 480)
        while not rospy.is_shutdown():
            if self.latest is not None:
                cv2.imshow('相机', self.bridge.imgmsg_to_cv2(self.latest, "bgr8"))
                cv2.waitKey(30)
            rospy.sleep(0.03)

    def take_photo(self):
        if self.latest is None: return
        path = f"{self.photo_dir}/photo_{self.photo_count:03d}.jpg"
        cv2.imwrite(path, self.bridge.imgmsg_to_cv2(self.latest, "bgr8"))
        print(f"📸 {path}")
        self.photo_count += 1

    def get_key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def run(self):
        import tty, termios
        threading.Thread(target=self.show_camera, daemon=True).start()
        print("="*40)
        print("  WASD移动 | 空格拍照 | Q退出")
        print("  (云台已移除，使用Husky原生RealSense D435)")
        print("="*40)
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            while not rospy.is_shutdown():
                key = self.get_key()
                twist = Twist()
                if key == 'w': twist.linear.x = 0.5
                elif key == 's': twist.linear.x = -0.5
                elif key == 'a': twist.angular.z = 1.0
                elif key == 'd': twist.angular.z = -1.0
                elif key == ' ':
                    self.take_photo()
                elif key == 'q':
                    break
                self.cmd_pub.publish(twist)
                rospy.sleep(0.05)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    RobotController().run()
