#!/usr/bin/env python3
"""里程计漂移矫正: 用 Gazebo 真值修正 odom 位置"""
import rospy, math, tf2_ros
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

class DriftCorrector:
    def __init__(self):
        rospy.init_node('drift_corrector')
        self.gt_x = self.gt_y = 0.0
        self.odom_x = self.odom_y = 0.0
        self.have_gt = self.have_odom = False

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gt_cb)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)

        self.br = tf2_ros.TransformBroadcaster()
        rospy.loginfo("✅ 漂移矫正已启动")

    def gt_cb(self, msg):
        try:
            i = msg.name.index('husky')
            self.gt_x = msg.pose[i].position.x
            self.gt_y = msg.pose[i].position.y
            self.have_gt = True
        except ValueError:
            pass

    def odom_cb(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.have_odom = True

        if self.have_gt:
            # 发布 map→odom 矫正变换
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = 'map'
            t.child_frame_id = 'odom'
            # 矫正量 = 真值 - odom
            t.transform.translation.x = self.gt_x - self.odom_x
            t.transform.translation.y = self.gt_y - self.odom_y
            t.transform.translation.z = 0.0
            t.transform.rotation.w = 1.0
            self.br.sendTransform(t)

if __name__ == '__main__':
    try:
        DriftCorrector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
