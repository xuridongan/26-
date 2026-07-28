#!/usr/bin/env python3
"""
navigation_controller.py

多点导航控制器：按顺序访问场地中的多个目标点，
到达后停驻让感知节点识别周围物体。
"""

import rospy
import actionlib
import math

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion
from std_msgs.msg import String
import tf
import tf2_ros


class NavigationController:
    def __init__(self):
        rospy.init_node('navigation_controller', anonymous=False)

        # 导航目标点序列 [x, y, yaw]
        # 在 4m×4m 场地中布置的关键检测点
        self.nav_goals = [
            # [x, y, yaw_rad]
            [0.0, 0.0, 0.0],               # P0: 起点（中心区域，人群检测）
            [1.2, 1.2, 0.785],              # P1: 东北（建筑+人群）
            [1.2, -0.95, -0.785],           # P2: 东南（垃圾桶）
            [-1.3, 1.0, 2.356],             # P3: 西北（建筑+人群）
            [-1.0, -1.2, -2.356],           # P4: 西南（建筑）
            [-1.3, -0.5, 3.14159],          # P5: 西侧（垃圾桶）
            [0.3, 0.4, 0.0],                # P6: 中心人群区
        ]

        self.current_goal_index = 0
        self.goal_reached = False

        # Action client for move_base
        self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("等待 move_base 服务器...")
        self.ac.wait_for_server()
        rospy.loginfo("move_base 服务器连接成功！")

        # TF listener
        self.tf_listener = tf.TransformListener()

        # Subscribe to detection results
        self.detection_sub = rospy.Subscriber('/detection/objects', String, self.detection_callback)

        rospy.loginfo("Navigation Controller 已启动，共 %d 个目标点", len(self.nav_goals))

    def detection_callback(self, msg):
        """收到检测结果时的回调"""
        rospy.loginfo("[Controller] 检测结果: %s", msg.data)

    def send_goal(self, x, y, yaw):
        """发送导航目标点"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position = Point(x, y, 0.0)

        # Convert yaw to quaternion
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation = Quaternion(*q)

        rospy.loginfo("=" * 50)
        rospy.loginfo("前往目标点 %d/%d: (%.2f, %.2f), yaw=%.2f",
                      self.current_goal_index + 1, len(self.nav_goals), x, y, yaw)
        rospy.loginfo("=" * 50)

        self.ac.send_goal(goal)

    def run(self):
        """主循环：依次访问所有目标点"""
        rate = rospy.Rate(1)

        # 依次导航到各目标点
        for i, goal_pose in enumerate(self.nav_goals):
            self.current_goal_index = i
            self.send_goal(goal_pose[0], goal_pose[1], goal_pose[2])

            # 等待到达目标
            finished = self.ac.wait_for_result(timeout=rospy.Duration(60.0))

            if finished:
                state = self.ac.get_state()
                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("✓ 已到达目标点 %d!", i + 1)
                    # 到达后停驻 3 秒让感知节点识别
                    rospy.loginfo("停驻检测周围物体...")
                    rospy.sleep(3.0)
                else:
                    rospy.logwarn("⚠ 目标点 %d 未完全到达 (状态: %d)", i + 1, state)
            else:
                rospy.logwarn("⚠ 目标点 %d 导航超时", i + 1)

        rospy.loginfo("*" * 50)
        rospy.loginfo("所有目标点访问完毕！")
        rospy.loginfo("*" * 50)

        # 完成后保持运行，等待手动停止
        rospy.spin()


if __name__ == '__main__':
    try:
        controller = NavigationController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation Controller 已停止")
