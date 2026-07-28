#!/usr/bin/env python3
"""多点巡航: 2D Nav Goal 设路径点（含朝向），到点后转向目标方向"""
import rospy, math, threading
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, Twist
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class Patrol:
    def __init__(self):
        rospy.init_node('patrol')
        self.pts = []; self.idx = 0; self.run = False

        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)
        rospy.Subscriber('/clicked_point', PointStamped, self.click_cb)
        rospy.Subscriber('/patrol_clear', Empty, self.clear_cb)
        rospy.Subscriber('/patrol_undo', Empty, self.undo_cb)

        self.mpub = rospy.Publisher('/patrol_markers', MarkerArray, queue_size=10)
        self.cpub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        threading.Thread(target=self.loop, daemon=True).start()
        rospy.loginfo("✅ 就绪！用 2D Nav Goal 设路径点（拖箭头设朝向）")

    def goal_cb(self, msg):
        self.pts.append(msg.pose)
        yaw = 2 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)
        rospy.loginfo(f"📌 {len(self.pts)}: ({msg.pose.position.x:.1f},{msg.pose.position.y:.1f}) {math.degrees(yaw):.0f}°")
        self.draw()
        if not self.run: self.run = True

    def click_cb(self, msg):
        p = msg.point; pose=Pose()
        pose.position.x, pose.position.y = p.x, p.y; pose.orientation.w = 1
        if self.pts:
            y = math.atan2(p.y-self.pts[-1].position.y, p.x-self.pts[-1].position.x)
            pose.orientation.z, pose.orientation.w = math.sin(y/2), math.cos(y/2)
        self.pts.append(pose)
        rospy.loginfo(f"📌 {len(self.pts)}: ({p.x:.1f},{p.y:.1f})")
        self.draw()
        if not self.run: self.run = True

    def clear_cb(self, _):
        self.pts=[]; self.idx=0; self.run=False; self.ac.cancel_all_goals(); self.draw(); rospy.loginfo("🗑️ 清除")

    def undo_cb(self, _):
        if self.pts:
            self.pts.pop(); rospy.loginfo("↩️")
            self.idx = min(self.idx, max(0, len(self.pts)-1))
            if not self.pts: self.run=False
            self.draw()

    def draw(self):
        ma = MarkerArray()
        c=Marker();c.action=Marker.DELETEALL;ma.markers.append(c)
        for i,p in enumerate(self.pts):
            s=Marker();s.header.frame_id='odom';s.header.stamp=rospy.Time.now()
            s.ns='p';s.id=i;s.type=Marker.SPHERE;s.action=Marker.ADD
            s.pose=p;s.pose.position.z=0.3;s.scale.x=0.35;s.scale.y=0.35;s.scale.z=0.35;s.color.a=0.8
            if i<self.idx: s.color.r=s.color.g=s.color.b=0.5
            elif i==self.idx: s.color.g=1
            else: s.color.b=1
            ma.markers.append(s)
            a=Marker();a.header.frame_id='odom';a.header.stamp=rospy.Time.now()
            a.ns='a';a.id=i;a.type=Marker.ARROW;a.action=Marker.ADD
            a.pose=p;a.pose.position.z=0.3;a.scale.x=0.5;a.scale.y=0.12;a.scale.z=0.12
            a.color.r=1;a.color.g=0.5;a.color.a=0.9
            ma.markers.append(a)
            if i>0:
                l=Marker();l.header.frame_id='odom';l.header.stamp=rospy.Time.now()
                l.ns='l';l.id=i;l.type=Marker.LINE_STRIP;l.action=Marker.ADD
                l.scale.x=0.04;l.color.r=l.color.g=1;l.color.a=0.4
                l.points=[self.pts[i-1].position,p.position]
                ma.markers.append(l)
        self.mpub.publish(ma)

    def rotate(self, pose):
        yaw = 2 * math.atan2(pose.orientation.z, pose.orientation.w)
        rospy.loginfo(f"🔄 转向 {math.degrees(yaw):.0f}°")
        t=Twist();t.angular.z=0.35
        for _ in range(20):
            self.cpub.publish(t);rospy.sleep(0.08)
        self.cpub.publish(Twist());rospy.sleep(0.5)

    def loop(self):
        while not rospy.is_shutdown():
            if not self.run or not self.pts:
                rospy.sleep(0.5);continue

            idx = self.idx % len(self.pts)
            g=MoveBaseGoal();g.target_pose.header.frame_id='odom'
            g.target_pose.pose=self.pts[idx]

            # 每次新 action client，避免状态冲突
            try:
                ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
                if not ac.wait_for_server(rospy.Duration(5)):
                    rospy.logwarn("move_base 不可用，重试")
                    rospy.sleep(2); continue
                ac.send_goal(g)
                ok = ac.wait_for_result(rospy.Duration(60))
                state = ac.get_state() if ok else -1
                ac.cancel_all_goals()
            except:
                state = -1

            if state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo(f"✅ {idx+1}")
                self.draw()
                self.rotate(self.pts[idx])
                self.idx += 1
            else:
                rospy.logwarn(f"⚠️ {idx+1} 超时/失败({state})，跳过")
                self.idx += 1

if __name__ == '__main__':
    try:
        Patrol()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
