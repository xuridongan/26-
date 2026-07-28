#!/usr/bin/env python3
"""用 Gazebo 真值动态发布 map→odom TF，消除里程计漂移"""
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
import math

class MapOdomPublisher:
    def __init__(self):
        rospy.init_node('map_odom_publisher')

        self.husky_world = None      # Gazebo 真值 (x, y, yaw)
        self.husky_odom = None       # 里程计 (x, y, yaw)
        self.last_publish = rospy.Time(0)

        # 订阅 Gazebo 真值
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_cb)
        # 订阅滤波里程计
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)

        self.br = tf2_ros.TransformBroadcaster()

        rospy.loginfo("📡 map→odom 动态发布器启动")

    def model_cb(self, msg):
        # 找 Husky 索引
        try:
            idx = msg.name.index('husky')
        except ValueError:
            return

        pose = msg.pose[idx]
        self.husky_world = (
            pose.position.x,
            pose.position.y,
            2 * math.atan2(pose.orientation.z, pose.orientation.w)
        )
        self.publish_tf()

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self.husky_odom = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            2 * math.atan2(q.z, q.w)
        )
        self.publish_tf()

    def publish_tf(self):
        if self.husky_world is None or self.husky_odom is None:
            return

        now = rospy.Time.now()

        # map→odom = Husky_world_pose - Husky_odom_pose
        # odom 原点在 map 中的位置 = 机器人在世界中的位置 - 机器人在 odom 中的位置
        dx = self.husky_world[0] - self.husky_odom[0]
        dy = self.husky_world[1] - self.husky_odom[1]
        dyaw = self.husky_world[2] - self.husky_odom[2]

        # 归一化
        while dyaw > math.pi: dyaw -= 2*math.pi
        while dyaw < -math.pi: dyaw += 2*math.pi

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = dx
        t.transform.translation.y = dy
        t.transform.translation.z = 0.0
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, dyaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.br.sendTransform(t)

if __name__ == '__main__':
    MapOdomPublisher()
    rospy.spin()
