#!/usr/bin/env python3
"""固定路线巡航: 沿预设路径点循环巡逻，指定点停靠转向"""
import rospy, math, threading
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# 路径点: (x, y, 朝向角度°)
# stop=True 的到点后旋转调整朝向，False 的直接路过
WAYPOINTS = [
    (4.8,  0.3,   69,  False),   # 1
    (6.4,  6.3,  177,  True),    # 2 → 停
    (6.5,  9.0,  175,  False),   # 2.5 过渡
    (6.6,  11.5, 175,  False),   # 3
    (4.0,  14.0, -90,  False),   # 3.5 过渡
    (0.5,  14.0, -80,  True),    # 4 → 停
    (-3.5,  14.0, -96,  False),   # 4.5 过渡
    (-6.0, 14.0, -96,  False),   # 5
    (-7.4,  7.5,  -2,  True),    # 6 → 停
    (-7.0,  2.0,  -5,  False),   # 7
    (-4.0,  1.0,  30,  False),   # 7.5 过渡
    (-0.7,  0.0,  88,  False),   # 8
]

class PatrolRoute:
    def __init__(self):
        rospy.init_node('patrol_route', anonymous=True)
        self.idx = 0
        self.pts = []

        for x, y, deg, stop in WAYPOINTS:
            p = Pose()
            p.position.x, p.position.y = x, y
            rad = math.radians(deg)
            p.orientation.z = math.sin(rad/2)
            p.orientation.w = math.cos(rad/2)
            self.pts.append((p, stop))

        self.mpub = rospy.Publisher('/patrol_markers', MarkerArray, queue_size=10)
        self.cpub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rx = self.ry = self.ryaw = 0.0
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)
        self.draw()

        threading.Thread(target=self.loop, daemon=True).start()
        rospy.loginfo(f"✅ 巡航就绪！{len(self.pts)} 个路径点，循环巡逻")

    def draw(self):
        ma = MarkerArray()
        c=Marker();c.action=Marker.DELETEALL;ma.markers.append(c)
        for i,(p,stop) in enumerate(self.pts):
            s=Marker()
            s.header.frame_id='odom';s.header.stamp=rospy.Time.now()
            s.ns='p';s.id=i;s.type=Marker.SPHERE;s.action=Marker.ADD
            s.pose=p;s.pose.position.z=0.3
            s.scale.x=0.35;s.scale.y=0.35;s.scale.z=0.35;s.color.a=0.8
            if i<self.idx: s.color.r=s.color.g=s.color.b=0.5
            elif i==self.idx: s.color.g=1
            else: s.color.b=1
            ma.markers.append(s)

            a=Marker()
            a.header.frame_id='odom';a.header.stamp=rospy.Time.now()
            a.ns='a';a.id=i;a.type=Marker.ARROW;a.action=Marker.ADD
            a.pose=p;a.pose.position.z=0.3
            a.scale.x=0.5;a.scale.y=0.12;a.scale.z=0.12
            a.color.r=1;a.color.g=0.5;a.color.a=0.9
            ma.markers.append(a)

            if i>0:
                l=Marker()
                l.header.frame_id='odom';l.header.stamp=rospy.Time.now()
                l.ns='l';l.id=i;l.type=Marker.LINE_STRIP;l.action=Marker.ADD
                l.scale.x=0.04;l.color.r=l.color.g=1;l.color.a=0.4
                l.points=[self.pts[i-1][0].position,p.position]
                ma.markers.append(l)

            # 标记停靠点
            if stop:
                t=Marker()
                t.header.frame_id='odom';t.header.stamp=rospy.Time.now()
                t.ns='stop';t.id=i;t.type=Marker.TEXT_VIEW_FACING;t.action=Marker.ADD
                t.pose=p;t.pose.position.z=1.0
                t.text="STOP";t.scale.z=0.5
                t.color.r=1;t.color.g=0;t.color.b=0;t.color.a=0.8
                ma.markers.append(t)

        self.mpub.publish(ma)

    def odom_cb(self, msg):
        self.rx = msg.pose.pose.position.x
        self.ry = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.ryaw = 2 * math.atan2(q.z, q.w)

    def rotate(self, pose):
        target = 2 * math.atan2(pose.orientation.z, pose.orientation.w)
        rospy.loginfo(f"🔄 转向 {math.degrees(target):.0f}°")

        for _ in range(120):
            err = target - self.ryaw
            while err > math.pi: err -= 2*math.pi
            while err < -math.pi: err += 2*math.pi
            if abs(err) < 0.02:
                break

            speed = abs(err) * 2.0
            if speed > 0.6: speed = 0.6
            if speed < 0.08: speed = 0.08

            t = Twist()
            t.angular.z = speed if err > 0 else -speed
            self.cpub.publish(t)
            rospy.sleep(0.05)

        self.cpub.publish(Twist())
        rospy.sleep(0.3)

    def set_tolerance(self, xy_tol, yaw_tol=None):
        """动态调整 DWA 容忍度"""
        try:
            rospy.set_param('/move_base/DWAPlannerROS/xy_goal_tolerance', xy_tol)
            if yaw_tol is not None:
                rospy.set_param('/move_base/DWAPlannerROS/yaw_goal_tolerance', yaw_tol)
        except:
            pass

    def loop(self):
        # 路过点用持久 ActionClient（连续切换不重建）
        ac_pass = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        ac_pass.wait_for_server(rospy.Duration(30))

        while not rospy.is_shutdown():
            idx = self.idx % len(self.pts)
            pose, stop = self.pts[idx]
            g=MoveBaseGoal();g.target_pose.header.frame_id='odom'
            g.target_pose.pose=pose

            try:
                if stop:
                    rospy.loginfo(f"🔴 停靠点 {idx+1} → ({pose.position.x:.1f},{pose.position.y:.1f})")
                    # 用跟路过点一样的方式导航到停靠点
                    self.set_tolerance(0.1, 0.5)
                    ac_pass.send_goal(g)
                    # 等待靠近到 0.2m 以内
                    for _ in range(300):
                        dist = math.hypot(pose.position.x-self.rx, pose.position.y-self.ry)
                        if dist < 0.2:
                            break
                        rospy.sleep(0.1)
                    # 取消导航，等 move_base 停止发指令
                    ac_pass.cancel_all_goals()
                    rospy.sleep(0.5)
                    self.cpub.publish(Twist())
                    rospy.sleep(0.2)
                    rospy.loginfo(f"✅ {idx+1} 到达停靠点，开始转向")
                    self.rotate(pose)
                    rospy.loginfo(f"🔄 {idx+1} 转向完成")
                    rospy.sleep(1)
                else:
                    # 路过点：1.5m 直接 preempt 去下个点
                    self.set_tolerance(0.3, 0.5)
                    ac_pass.send_goal(g)
                    while not rospy.is_shutdown():
                        dist = math.hypot(pose.position.x-self.rx, pose.position.y-self.ry)
                        s = ac_pass.get_state()
                        if dist < 1.5 or s in (actionlib.GoalStatus.SUCCEEDED, actionlib.GoalStatus.ABORTED):
                            break
                        rospy.sleep(0.1)
                    rospy.loginfo(f"➡️ {idx+1}")

                self.draw()
            except:
                pass
            self.idx += 1

if __name__ == '__main__':
    try:
        PatrolRoute()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
