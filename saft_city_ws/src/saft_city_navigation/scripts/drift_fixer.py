#!/usr/bin/env python3
"""用 Gazebo 真值定期修正 EKF 位置，消除里程计漂移"""
import rospy, math
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty

class DriftFixer:
    def __init__(self):
        # 每 30 秒修正一次（避免频繁跳变，也防止漂移积累太多）
        self.interval = 30.0  # 秒
        self.last_fix = 0
        self.husky_world = None
        self.set_pose = None

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb)
        rospy.loginfo(f"📡 漂移修正器启动，每 {self.interval}s 修正一次")

    def cb(self, msg):
        try:
            idx = msg.name.index('husky')
        except ValueError:
            return

        now = rospy.Time.now().to_sec()
        if now - self.last_fix < self.interval:
            return

        p = msg.pose[idx]

        # Gazebo world → odom frame
        # map→odom = (0.7, -7.0), so P_odom = P_world - (0.7, -7.0)
        x_in_odom = p.position.x - 0.7
        y_in_odom = p.position.y + 7.0

        # 只在大偏移时才修正（超过 0.3m）
        if math.hypot(x_in_odom, y_in_odom) < 0.3:
            return

        # 调用 EKF 的 set_pose 服务
        if self.set_pose is None:
            try:
                from rosgraph_msgs.msg import Clock
                rospy.wait_for_service('/set_pose', timeout=2)
                from robot_localization.srv import SetPose
                self.set_pose = rospy.ServiceProxy('/set_pose', SetPose)
            except:
                rospy.logwarn("⚠️ /set_pose 不可用，跳过修正")
                self.last_fix = now
                return

        try:
            req = PoseWithCovarianceStamped()
            req.header.stamp = rospy.Time.now()
            req.header.frame_id = 'odom'
            req.pose.pose.position.x = x_in_odom
            req.pose.pose.position.y = y_in_odom
            req.pose.pose.position.z = 0.0
            req.pose.pose.orientation = p.orientation
            # 给一个合适的协方差
            cov = [0.0]*36
            cov[0] = 0.005  # x^2 (7cm)
            cov[7] = 0.005  # y^2
            cov[35] = 0.05  # yaw^2
            req.pose.covariance = cov

            self.set_pose(req)
            rospy.loginfo(f"🔄 漂移修正: odom({x_in_odom:.2f},{y_in_odom:.2f})")
        except Exception as e:
            rospy.logwarn(f"⚠️ 修正失败: {e}")

        self.last_fix = now

if __name__ == '__main__':
    rospy.init_node('drift_fixer')
    DriftFixer()
    rospy.spin()
