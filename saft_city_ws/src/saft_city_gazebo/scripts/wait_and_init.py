#!/usr/bin/env python3
"""等待 Husky 就绪后加载控制器"""
import rospy, time, sys, subprocess
from gazebo_msgs.srv import GetModelProperties

rospy.init_node('init_controllers')
rospy.loginfo("等待机器人就绪...")

# 等待 Husky 在 Gazebo 中出现
try:
    rospy.wait_for_service('/gazebo/get_model_properties', timeout=60)
    get_props = rospy.ServiceProxy('/gazebo/get_model_properties', GetModelProperties)
    for i in range(30):
        try:
            resp = get_props('husky')
            if resp.body_names:
                rospy.loginfo("✅ Husky 就绪")
                break
        except:
            pass
        time.sleep(2)
except:
    rospy.logwarn("超时等待 Husky，继续尝试加载控制器")

# 加载控制器
rospy.loginfo("加载控制器...")
subprocess.run(['rosparam', 'load', '/opt/ros/noetic/share/husky_control/config/control.yaml'],
               capture_output=True)
for ctrl in ['husky_joint_publisher', 'husky_velocity_controller']:
    subprocess.run(['rosservice', 'call', '/controller_manager/load_controller',
                    f"name: '{ctrl}'"], capture_output=True)
subprocess.run(['rosservice', 'call', '/controller_manager/switch_controller',
                "{start_controllers: ['husky_joint_publisher', 'husky_velocity_controller'], "
                "stop_controllers: [], strictness: 2}"], capture_output=True)

rospy.loginfo("✅ 控制器加载完成")
