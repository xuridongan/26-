#!/usr/bin/env python3
"""机器人回到起点 (0.7, -7)"""
import rospy, actionlib
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

rospy.init_node('go_home')
ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
if not ac.wait_for_server(rospy.Duration(5)):
    rospy.logerr("move_base 不可用")
    exit(1)

goal = MoveBaseGoal()
goal.target_pose.header.frame_id = 'odom'
goal.target_pose.pose.position.x = 0.7
goal.target_pose.pose.position.y = -7.0
goal.target_pose.pose.orientation.w = 1.0

rospy.loginfo("➡️ 返回起点 (0.7, -7)...")
ac.send_goal(goal)
ac.wait_for_result(rospy.Duration(60))

if ac.get_state() == actionlib.GoalStatus.SUCCEEDED:
    rospy.loginfo("✅ 已回到起点")
else:
    rospy.logwarn(f"⚠️ 返回失败 ({ac.get_state()})")
