#!/usr/bin/env python3
"""流服务端 — 发原图→Windows, 收标注图→显示"""
import rospy, cv2, socket, threading, struct, pickle
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

HOST = '0.0.0.0'
PORT = 8888

class StreamServer:
    def __init__(self):
        rospy.init_node('stream_server')
        self.bridge = CvBridge()
        self.latest = None

        rospy.Subscriber('/realsense/color/image_raw', Image, self.img_cb, queue_size=1)
        self.result_pub = rospy.Publisher('/detection/result_image', Image, queue_size=10)

        threading.Thread(target=self.server_thread, daemon=True).start()
        rospy.loginfo(f"流服务端: {HOST}:{PORT} (发原图→收标注→显示)")

    def img_cb(self, msg):
        self.latest = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def server_thread(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        while not rospy.is_shutdown():
            try:
                conn, addr = s.accept()
                rospy.loginfo(f"主电脑已连接: {addr}")
                while not rospy.is_shutdown():
                    if self.latest is not None:
                        # 发原图
                        data = pickle.dumps(self.latest)
                        conn.sendall(struct.pack('>I', len(data)) + data)

                        # 收标注图（4字节大小 + 图片数据）
                        size_data = conn.recv(4)
                        if len(size_data) < 4: break
                        size = struct.unpack('>I', size_data)[0]
                        buf = b''
                        while len(buf) < size:
                            chunk = conn.recv(min(size - len(buf), 65536))
                            if not chunk: break
                            buf += chunk
                        if len(buf) >= size:
                            result = pickle.loads(buf)
                            # 发布标注图到ROS，robot_controller会显示
                            msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
                            self.result_pub.publish(msg)
                    rospy.sleep(0.03)
            except:
                rospy.sleep(1)

if __name__ == '__main__':
    StreamServer()
    rospy.spin()
