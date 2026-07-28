#!/usr/bin/env python3
"""启动时捕捉机器人位置，设 map→odom 固定偏移（静态TF）"""
import rospy, sys, subprocess
from gazebo_msgs.msg import ModelStates

rospy.init_node('set_initial_offset', anonymous=True)
rospy.loginfo("等待 Gazebo 获取初始位置...")

msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=30)
try:
    i = msg.name.index('husky')
    x = msg.pose[i].position.x
    y = msg.pose[i].position.y
    rospy.loginfo(f"初始位置: ({x:.2f}, {y:.2f}) → 设为 map→odom 偏移")

    # 杀光旧的 static_transform_publisher
    subprocess.run("rosnode list | grep static | xargs rosnode kill 2>/dev/null", shell=True)
    import time; time.sleep(1)

    # 用静态变换固定偏移（不更新，避免冲突）
    cmd = f"rosrun tf static_transform_publisher {x} {y} 0 0 0 0 map odom 10"
    subprocess.Popen(cmd, shell=True)
    rospy.loginfo("✅ 初始偏移已设置！RViz 中 Fixed Frame 改 map 即可对齐")
    rospy.spin()
except ValueError:
    rospy.logerr("找不到 husky 模型")
