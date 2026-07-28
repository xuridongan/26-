#!/usr/bin/env python3
"""生成环形跑道 world 文件"""
import math

R_OUTER = 3.5   # 外圈半径
R_INNER = 2.0   # 内圈半径
N_OUTER = 36    # 外圈柱子数量
N_INNER = 24    # 内圈柱子数量
POST_RADIUS = 0.04
POST_HEIGHT = 0.3

lines = []

lines.append('<?xml version="1.0"?>')
lines.append('<sdf version="1.6">')
lines.append('  <world name="circular_track_world">')
lines.append('    <physics type="ode">')
lines.append('      <real_time_update_rate>500</real_time_update_rate>')
lines.append('      <max_step_size>0.002</max_step_size>')
lines.append('    </physics>')
lines.append('    <include><uri>model://sun</uri></include>')
lines.append('    <include><uri>model://ground_plane</uri></include>')

# 外圈
for i in range(N_OUTER):
    angle = 2 * math.pi * i / N_OUTER
    x = R_OUTER * math.cos(angle)
    y = R_OUTER * math.sin(angle)
    lines.append(f'''    <model name="outer_post_{i}">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <pose>{x:.3f} {y:.3f} {POST_HEIGHT/2} 0 0 0</pose>
          <geometry><cylinder radius="{POST_RADIUS}" length="{POST_HEIGHT}"/></geometry>
        </collision>
        <visual name="visual">
          <pose>{x:.3f} {y:.3f} {POST_HEIGHT/2} 0 0 0</pose>
          <geometry><cylinder radius="{POST_RADIUS}" length="{POST_HEIGHT}"/></geometry>
          <material><ambient>0.2 0.2 0.7 1</ambient></material>
        </visual>
      </link>
    </model>''')

# 内圈
for i in range(N_INNER):
    angle = 2 * math.pi * i / N_INNER
    x = R_INNER * math.cos(angle)
    y = R_INNER * math.sin(angle)
    lines.append(f'''    <model name="inner_post_{i}">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <pose>{x:.3f} {y:.3f} {POST_HEIGHT/2} 0 0 0</pose>
          <geometry><cylinder radius="{POST_RADIUS}" length="{POST_HEIGHT}"/></geometry>
        </collision>
        <visual name="visual">
          <pose>{x:.3f} {y:.3f} {POST_HEIGHT/2} 0 0 0</pose>
          <geometry><cylinder radius="{POST_RADIUS}" length="{POST_HEIGHT}"/></geometry>
          <material><ambient>0.7 0.2 0.2 1</ambient></material>
        </visual>
      </link>
    </model>''')

lines.append('  </world>')
lines.append('</sdf>')

with open('/home/xuan/saft_city_ws/src/saft_city_gazebo/worlds/circular_track.world', 'w') as f:
    f.write('\n'.join(lines))

print(f"生成完成: 外圈{N_OUTER}根 + 内圈{N_INNER}根 = {N_OUTER+N_INNER}根圆柱柱子")
