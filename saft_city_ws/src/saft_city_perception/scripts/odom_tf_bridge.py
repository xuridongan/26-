#!/usr/bin/env python3
"""Odom → base_link TF桥（为SLAM提供TF连接）"""
import rospy, tf2_ros, math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

class OdomTFBridge:
    def __init__(self):
        rospy.init_node('odom_tf_bridge')
        self.br = tf2_ros.TransformBroadcaster()
        rospy.Subscriber('/husky_velocity_controller/odom', Odometry, self.odom_cb)
        rospy.loginfo("Odom TF Bridge started")

    def odom_cb(self, msg):
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = 0
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)

if __name__ == '__main__':
    OdomTFBridge()
    rospy.spin()
