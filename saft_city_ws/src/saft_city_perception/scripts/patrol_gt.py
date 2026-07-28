#!/usr/bin/env python3
"""精确巡航: Gazebo 真值定位 + 沿墙走 + 到站转向+停止"""
import rospy, math
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray

# 停靠点: (x, y, 朝向角度°) - 到站后转向此方向，停2秒
STOPS = [
    (6.4, 6.3,  177),   # 2
    (0.3, 14.2, -96),   # 4
    (-7.0, 10.3, -1),   # 6
    (-7.4, 5.7, -2),    # 7
]

class PatrolGT:
    def __init__(self):
        rospy.init_node('patrol_gt')
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.mpub = rospy.Publisher('/patrol_markers', MarkerArray, queue_size=10)
        rospy.Subscriber('/front/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gt_cb)

        self.px, self.py, self.pyaw = 0.0, 0.0, 0.0
        self.stop_idx = 0
        self.state = 'drive'  # drive / rotate / wait
        self.state_start = rospy.Time.now()
        self.target_yaw = 0.0
        self.draw_stops()
        rospy.loginfo(f"✅ 巡航启动！{len(STOPS)} 个停靠点")

    def gt_cb(self, msg):
        try:
            i = msg.name.index('husky')
            p = msg.pose[i]
            self.px = p.position.x
            self.py = p.position.y
            q = p.orientation
            self.pyaw = math.atan2(2*(q.w*q.z), 1-2*(q.z*q.z))
        except ValueError:
            pass

    def scan_cb(self, msg):
        t = Twist()

        if self.state == 'drive':
            # 检查是否到站
            if self.stop_idx < len(STOPS):
                tx, ty, angle = STOPS[self.stop_idx]
                dist = math.sqrt((tx-self.px)**2 + (ty-self.py)**2)
                if dist < 1.0:
                    rospy.loginfo(f"🛑 到站 {self.stop_idx+1}，转向 {angle}°")
                    self.target_yaw = math.radians(angle)
                    self.state = 'rotate'
                    self.state_start = rospy.Time.now()
                    self.cmd_pub.publish(Twist())
                    return

            # 沿墙走（跟右墙，逆时针绕圈）
            n = len(msg.ranges)
            front = min(msg.ranges[n//2], 5.0) if n > 0 else 5.0
            right = min(min(msg.ranges[int(n*0.6):int(n*0.8)]), 5.0) if n > 10 else 5.0

            if front < 0.5:
                t.linear.x = 0.05
                t.angular.z = 0.4
            else:
                # 保持在距右墙 0.6m
                error = 0.6 - right
                t.linear.x = 0.3
                t.angular.z = max(-0.3, min(0.3, error * 0.3))
            self.cmd_pub.publish(t)

        elif self.state == 'rotate':
            # 转向目标角度（闭环控制）
            err = self.target_yaw - self.pyaw
            # 规范化到 -pi ~ pi
            while err > math.pi: err -= 2*math.pi
            while err < -math.pi: err += 2*math.pi
            if abs(err) < 0.05:  # 误差 < 3°
                rospy.loginfo(f"✅ 方向到位")
                self.cmd_pub.publish(Twist())
                self.state = 'wait'
                self.state_start = rospy.Time.now()
                self.stop_idx += 1
                if self.stop_idx >= len(STOPS):
                    self.stop_idx = 0
                    rospy.loginfo("🔄 新一圈")
                return
            t.angular.z = max(-0.5, min(0.5, err * 2))
            self.cmd_pub.publish(t)

        elif self.state == 'wait':
            # 停2秒
            if (rospy.Time.now() - self.state_start).to_sec() > 2.0:
                self.state = 'drive'
                rospy.loginfo("➡️ 继续巡航")
            else:
                self.cmd_pub.publish(Twist())

    def draw_stops(self):
        ma = MarkerArray()
        c = Marker(); c.action = Marker.DELETEALL; ma.markers.append(c)
        for i, (x, y, angle) in enumerate(STOPS):
            s = Marker()
            s.header.frame_id = 'odom'; s.header.stamp = rospy.Time.now()
            s.ns = 'stops'; s.id = i; s.type = Marker.SPHERE
            s.action = Marker.ADD
            s.pose.position.x = x; s.pose.position.y = y; s.pose.position.z = 0.3
            s.scale.x = 0.5; s.scale.y = 0.5; s.scale.z = 0.5
            s.color.r = 1; s.color.g = 0; s.color.b = 0; s.color.a = 0.8
            ma.markers.append(s)
        self.mpub.publish(ma)

if __name__ == '__main__':
    try:
        PatrolGT()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
