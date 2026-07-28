#!/usr/bin/env python3
"""把 Gazebo 真值作为绝对位置输入给 EKF，消除里程计漂移"""
import rospy, math
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates

class GazeboPosePublisher:
    def __init__(self):
        self.pub = rospy.Publisher('/gazebo_pose', PoseWithCovarianceStamped, queue_size=10)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb)
        rospy.loginfo("📡 Gazebo 真值发布器启动 → /gazebo_pose")

    def cb(self, msg):
        try:
            idx = msg.name.index('husky')
        except ValueError:
            return

        p = msg.pose[idx]
        now = rospy.Time.now()

        out = PoseWithCovarianceStamped()
        out.header.stamp = now
        out.header.frame_id = 'odom'

        # Gazebo world → odom frame
        # map→odom = (0.7, -7.0), so P_odom = P_world - (0.7, -7.0)
        out.pose.pose.position.x = p.position.x - 0.7
        out.pose.pose.position.y = p.position.y + 7.0
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation = p.orientation

        # 协方差: 信任位置 (0.1m), 信任朝向 (0.1rad)
        # 格式: 6x6 row-major [x, y, z, roll, pitch, yaw]
        cov = [0.0] * 36
        cov[0] = 0.01   # x^2
        cov[7] = 0.01   # y^2
        cov[14] = 1.0   # z^2 (不关心)
        cov[21] = 1.0   # roll^2
        cov[28] = 1.0   # pitch^2
        cov[35] = 0.05  # yaw^2
        out.pose.covariance = cov

        self.pub.publish(out)

if __name__ == '__main__':
    rospy.init_node('gazebo_pose_publisher')
    GazeboPosePublisher()
    rospy.spin()
