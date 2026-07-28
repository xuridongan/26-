#!/usr/bin/env python3
"""激光里程计: 扫描匹配矫正里程计漂移，发布矫正后的odom"""
import rospy, math, numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros

class LaserOdom:
    def __init__(self):
        rospy.init_node('laser_odom')
        self.prev_pts = None
        self.cumul_x = self.cumul_y = 0.0
        self.robot_x = self.robot_y = 0.0
        self.first_odom = True
        self.base_ox = self.base_oy = 0.0

        rospy.Subscriber('/front/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)
        self.br = tf2_ros.TransformBroadcaster()
        rospy.loginfo("✅ 激光里程计启动")

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        if self.first_odom:
            self.base_ox = self.robot_x
            self.base_oy = self.robot_y
            self.first_odom = False

    def scan_to_pts(self, msg):
        pts = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                pts.append([r * math.cos(angle), r * math.sin(angle)])
            angle += msg.angle_increment
        return np.array(pts) if len(pts) > 50 else None

    def match(self, curr, prev):
        """ICP匹配：找两帧间位移"""
        if curr is None or prev is None or len(curr) < 20 or len(prev) < 20:
            return 0, 0
        cx, cy = curr[::3, 0], curr[::3, 1]  # 降采样
        px, py = prev[::3, 0], prev[::3, 1]
        dx, dy, count = 0, 0, 0
        for i in range(len(cx)):
            dists = (px - cx[i])**2 + (py - cy[i])**2
            j = np.argmin(dists)
            if dists[j] < 0.5:
                dx += cx[i] - px[j]
                dy += cy[i] - py[j]
                count += 1
        return (dx/count, dy/count) if count > 10 else (0, 0)

    def scan_cb(self, msg):
        curr = self.scan_to_pts(msg)
        if self.prev_pts is not None and not self.first_odom:
            dx, dy = self.match(curr, self.prev_pts)
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                self.cumul_x += dx
                self.cumul_y += dy
                # 发布矫正后的odom变换
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = 'odom'
                t.child_frame_id = 'base_link'
                # 融合scan matching和里程计
                t.transform.translation.x = self.robot_x + dx
                t.transform.translation.y = self.robot_y + dy
                t.transform.translation.z = 0.0
                t.transform.rotation.w = 1.0
                self.br.sendTransform(t)
        self.prev_pts = curr

if __name__ == '__main__':
    try:
        LaserOdom()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
