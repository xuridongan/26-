#!/usr/bin/env python3
"""显示路径点标记（独立脚本，不受旧节点干扰）"""
import rospy, math
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker, MarkerArray

WAYPOINTS = [
    (4.8,  0.3,   69),   # 1
    (6.4,  6.3,  177),   # 2
    (6.6,  12.7, 175),   # 3
    (0.3,  14.2, -96),   # 4
    (-6.3, 12.8, -96),   # 5
    (-7.0, 10.3,  -1),   # 6
    (-7.4,  5.7,  -2),   # 7
    (-6.2,  1.1,  -5),   # 8
    (-0.7,  0.0,  88),   # 9
]

rospy.init_node('waypoint_display')
pub = rospy.Publisher('/waypoints_display', MarkerArray, queue_size=10, latch=True)
rospy.sleep(0.5)

ma = MarkerArray()
for i, (x, y, deg) in enumerate(WAYPOINTS):
    p = Pose()
    p.position.x, p.position.y = x, y
    rad = math.radians(deg)
    p.orientation.z = math.sin(rad/2)
    p.orientation.w = math.cos(rad/2)

    # 球体
    s = Marker()
    s.header.frame_id = 'odom'; s.header.stamp = rospy.Time.now()
    s.ns = 'pts'; s.id = i; s.type = Marker.SPHERE
    s.action = Marker.ADD; s.pose = p; s.pose.position.z = 0.3
    s.scale.x = 0.4; s.scale.y = 0.4; s.scale.z = 0.4
    s.color.b = 1; s.color.g = 0.5; s.color.a = 0.8
    ma.markers.append(s)

    # 箭头
    a = Marker()
    a.header.frame_id = 'odom'; a.header.stamp = rospy.Time.now()
    a.ns = 'arr'; a.id = i; a.type = Marker.ARROW
    a.action = Marker.ADD; a.pose = p; a.pose.position.z = 0.3
    a.scale.x = 0.6; a.scale.y = 0.15; a.scale.z = 0.15
    a.color.r = 1; a.color.g = 0.5; a.color.a = 0.9
    ma.markers.append(a)

    # 连线
    if i > 0:
        l = Marker()
        l.header.frame_id = 'odom'; l.header.stamp = rospy.Time.now()
        l.ns = 'ln'; l.id = i; l.type = Marker.LINE_STRIP
        l.action = Marker.ADD; l.scale.x = 0.04
        l.color.r = 1; l.color.g = 1; l.color.a = 0.4
        px, py = WAYPOINTS[i-1][0], WAYPOINTS[i-1][1]
        l.points = [Pose().position for _ in range(2)]
        l.points[0].x = px; l.points[0].y = py
        l.points[1].x = x; l.points[1].y = y
        ma.markers.append(l)

pub.publish(ma)
rospy.loginfo(f"✅ 已发布 {len(WAYPOINTS)} 个路径点")
rospy.spin()
