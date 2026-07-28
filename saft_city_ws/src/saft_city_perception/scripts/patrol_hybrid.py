#!/usr/bin/env python3
"""混合巡航: 沿墙走（不漂移）+ 到点停靠转向"""
import rospy, math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray

# 停靠点: (跑多少米后停, 转向角度)
# 赛道一圈约 42m，9 个点均匀分布
STOPS = [
    (0,    None),   # 1: 起点，不停
    (4.7,  177),    # 2: 停靠 177°
    (9.3,  None),   # 3: 路过
    (14.0, -96),    # 4: 停靠 -96°
    (18.7, None),   # 5: 路过
    (23.3, -1),     # 6: 停靠 -1°
    (28.0, -2),     # 7: 停靠 -2°
    (32.7, None),   # 8: 路过
    (37.3, None),   # 9: 路过
]

class PatrolHybrid:
    def __init__(self):
        rospy.init_node('patrol_hybrid')
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.mpub = rospy.Publisher('/patrol_markers', MarkerArray, queue_size=10)
        rospy.Subscriber('/front/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)

        self.target_dist = 0.8
        self.max_speed = 0.5
        self.x0 = None        # 起点 x
        self.dist = 0.0       # 已走距离
        self.stop_idx = 0     # 当前目标停靠点索引
        self.stop_state = 0   # 0=沿墙走, 1=停靠转向中
        self.lap = 0

        self.publish_markers()
        rospy.loginfo("✅ 混合巡航启动！%d 个停靠点", len(STOPS))

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.x0 is None:
            self.x0, self.y0 = x, y
        # 计算从起点走的直线距离（近似弧长）
        self.dist = math.sqrt((x - self.x0)**2 + (y - self.y0)**2)

    def scan_cb(self, msg):
        t = Twist()
        if self.stop_state == 1:
            self.cmd_pub.publish(t)
            return  # 停靠中，不动

        n = len(msg.ranges)
        front = min(min(msg.ranges[n//2-10:n//2+10]), 5.0) if n > 20 else min(msg.ranges[n//2], 5.0)
        left  = min(min(msg.ranges[n//4-5:n//4+5]), 5.0) if n > 10 else min(msg.ranges[n//4], 5.0)
        right = min(min(msg.ranges[3*n//4-5:3*n//4+5]), 5.0) if n > 10 else min(msg.ranges[3*n//4], 5.0)

        # 检查是否到达停靠点
        if self.stop_idx < len(STOPS) and self.dist >= STOPS[self.stop_idx][0]:
            angle = STOPS[self.stop_idx][1]
            if angle is not None:
                self.stop_state = 1
                rospy.loginfo(f"🛑 到站 {self.stop_idx+1}，转向 {angle}°")
                self.rotate(angle)
                self.stop_state = 0
            self.stop_idx += 1
            if self.stop_idx >= len(STOPS):
                self.stop_idx = 0
                self.lap += 1
                self.x0, self.y0 = None, None  # 重置里程计基准
                rospy.loginfo(f"🔄 第 {self.lap+1} 圈")
            return

        # 沿墙走
        if front < 0.6:
            t.linear.x = 0.1
            t.angular.z = 0.6
        else:
            error = self.target_dist - min(left, right)
            t.linear.x = self.max_speed
            t.angular.z = max(-0.6, min(0.6, -error * 0.5))

        self.cmd_pub.publish(t)

    def rotate(self, target_deg):
        rospy.loginfo(f"🔄 转向 {target_deg}°")
        t = Twist()
        t.angular.z = 0.3
        for _ in range(25):
            self.cmd_pub.publish(t)
            rospy.sleep(0.08)
        self.cmd_pub.publish(Twist())
        rospy.sleep(2)

    def publish_markers(self):
        ma = MarkerArray()
        c = Marker(); c.action = Marker.DELETEALL; ma.markers.append(c)
        for i, (d, angle) in enumerate(STOPS):
            s = Marker()
            s.header.frame_id = 'odom'; s.header.stamp = rospy.Time.now()
            s.ns = 'stops'; s.id = i; s.type = Marker.SPHERE
            s.action = Marker.ADD; s.scale.x = 0.3; s.scale.y = 0.3; s.scale.z = 0.3
            s.color.g = 1; s.color.a = 0.7
            s.pose.position.x = d * 0  # 没法画在路径上
            s.pose.position.y = 0
            ma.markers.append(s)
        self.mpub.publish(ma)

if __name__ == '__main__':
    try:
        PatrolHybrid()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
