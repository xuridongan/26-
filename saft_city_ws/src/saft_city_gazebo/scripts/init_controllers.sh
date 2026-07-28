#!/bin/bash
# 等待机器人就绪，然后加载控制器
# 由 mapping.launch 自动调用

# 等待 odom 话题（证明控制器在跑）
for i in $(seq 1 30); do
  if rostopic list 2>/dev/null | grep -q "/husky_velocity_controller/odom"; then
    break
  fi
  sleep 2
done

# 检查机器人模型是否在 Gazebo 中
for i in $(seq 1 15); do
  result=$(rosservice call /gazebo/get_model_properties "{model_name: 'husky'}" 2>/dev/null)
  if echo "$result" | grep -q "base_link"; then
    break
  fi
  sleep 2
done

# 加载控制器（幂等，已存在会报错但不影响）
rosparam load /opt/ros/noetic/share/husky_control/config/control.yaml 2>/dev/null
rosservice call /controller_manager/load_controller "name: 'husky_joint_publisher'" 2>/dev/null
rosservice call /controller_manager/load_controller "name: 'husky_velocity_controller'" 2>/dev/null
rosservice call /controller_manager/switch_controller "{start_controllers: ['husky_joint_publisher', 'husky_velocity_controller'], stop_controllers: [], strictness: 2}" 2>/dev/null

# 启动 robot_state_publisher
rosrun robot_state_publisher robot_state_publisher __name:=rsp_auto &
