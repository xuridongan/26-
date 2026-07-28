#!/usr/bin/env python3
"""用 Gazebo 真值对齐 map→odom，让 RViz 地图和实际位置一致"""
import rospy, tf2_ros
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

rospy.init_node('map_align')
br = tf2_ros.TransformBroadcaster()
gt_x = gt_y = 0.0
odom_x = odom_y = 0.0
got_gt = got_odom = False

def gt_cb(msg):
    global gt_x, gt_y, got_gt
    try:
        i = msg.name.index('husky')
        gt_x = msg.pose[i].position.x
        gt_y = msg.pose[i].position.y
        got_gt = True
        publish()
    except ValueError:
        pass

def odom_cb(msg):
    global odom_x, odom_y, got_odom
    odom_x = msg.pose.pose.position.x
    odom_y = msg.pose.pose.position.y
    got_odom = True
    publish()

def publish():
    if not (got_gt and got_odom):
        return
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'map'
    t.child_frame_id = 'odom'
    # map→odom = 真值 - odom值（矫正漂移）
    t.transform.translation.x = gt_x - odom_x
    t.transform.translation.y = gt_y - odom_y
    t.transform.translation.z = 0.0
    t.transform.rotation.w = 1.0
    br.sendTransform(t)

rospy.Subscriber('/gazebo/model_states', ModelStates, gt_cb)
rospy.Subscriber('/odometry/filtered', Odometry, odom_cb)
rospy.loginfo("✅ RViz 地图对齐已启动！RViz 中 Fixed Frame 设为 map")
rospy.spin()
