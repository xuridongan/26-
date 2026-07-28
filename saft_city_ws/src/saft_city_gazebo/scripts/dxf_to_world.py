#!/usr/bin/env python3
"""将 pingan_city DXF 文件转换为 Gazebo SDF world"""
import re, math

DXF_PATH = '/home/xuan/桌面/pingan_city_zwcad_formal.dxf'
OUT_PATH = '/home/xuan/saft_city_ws/src/saft_city_gazebo/worlds/circular_track.world'

with open(DXF_PATH, 'r') as f:
    content = f.read()

# ---- 解析 DXF ----
lines = []
arcs = []
circles = []
polylines = []

# LINE: 10,20 -> start; 11,21 -> end
for m in re.finditer(r'0\nLINE\n(?:.*?\n)*?10\n([\d.-]+)\n20\n([\d.-]+)\n(?:.*?\n)*?11\n([\d.-]+)\n21\n([\d.-]+)', content):
    lines.append(((float(m.group(1)), float(m.group(2))), (float(m.group(3)), float(m.group(4)))))

# CIRCLE: 10,20 -> center; 40 -> radius
for m in re.finditer(r'0\nCIRCLE\n(?:.*?\n)*?10\n([\d.-]+)\n20\n([\d.-]+)\n(?:.*?\n)*?40\n([\d.-]+)', content):
    circles.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))

# ARC: 10,20 -> center; 40 -> radius; 50 -> start angle; 51 -> end angle
for m in re.finditer(r'0\nARC\n(?:.*?\n)*?10\n([\d.-]+)\n20\n([\d.-]+)\n(?:.*?\n)*?40\n([\d.-]+)\n(?:.*?\n)*?50\n([\d.-]+)\n(?:.*?\n)*?51\n([\d.-]+)', content):
    arcs.append((float(m.group(1)), float(m.group(2)), float(m.group(3)),
                 float(m.group(4)), float(m.group(5))))

# POLYLINE -> VERTEX
for ps in content.split('0\nPOLYLINE')[1:]:
    verts = re.findall(r'10\n([\d.-]+)\n20\n([\d.-]+)', ps)
    if verts:
        poly = [(float(v[0]), float(v[1])) for v in verts]
        polylines.append(poly)

# ---- 计算中心偏移 ----
all_pts = []
for (x1,y1),(x2,y2) in lines:
    all_pts.extend([(x1,y1),(x2,y2)])
for cx,cy,r in circles:
    all_pts.append((cx,cy))
for cx,cy,r,sa,ea in arcs:
    all_pts.append((cx,cy))
for poly in polylines:
    all_pts.extend(poly)

if not all_pts:
    print("ERROR: No entities found!")
    exit(1)

xs = [p[0] for p in all_pts]
ys = [p[1] for p in all_pts]
cx = (max(xs) + min(xs)) / 2
cy = (max(ys) + min(ys)) / 2
print(f"DXF 中心: ({cx:.1f}, {cy:.1f})")
print(f"实体: {len(lines)}条线, {len(arcs)}个弧, {len(circles)}个圆, {len(polylines)}条多段线")

# 计算比例: DXF是mm, 目标±2m (4m场地)
max_extent = max(max(xs)-min(xs), max(ys)-min(ys))
scale = 4.0 / max_extent  # mm -> m, 且轨道外径<=4m
print(f"原始范围: {max_extent:.0f}mm, SDF比例: {scale:.6f} (→±{max_extent*scale/2:.2f}m)")

# ---- 过滤: 只保留赛道区域(排除图纸说明表格和图框) ----
# 赛道中心区域 DXF 坐标范围
TRACK_MIN_X, TRACK_MAX_X = 200, 3800
TRACK_MIN_Y, TRACK_MAX_Y = 200, 3800

def in_track_area(x, y):
    return TRACK_MIN_X <= x <= TRACK_MAX_X and TRACK_MIN_Y <= y <= TRACK_MAX_Y

# ---- 生成 SDF world ----
WALL_HEIGHT = 0.15    # 赛道围栏高度
WALL_THICK = 0.06     # 围栏厚度
MIN_LENGTH = 30       # 过滤短线段(图纸标注文字), 单位: DXF mm

sdf = []
sdf.append('<?xml version="1.0"?>')
sdf.append('<sdf version="1.6">')
sdf.append('  <world name="saft_city_arena">')
sdf.append('    <include><uri>model://sun</uri></include>')
sdf.append('    <include><uri>model://ground_plane</uri></include>')
sdf.append('    <physics type="ode"><real_time_update_rate>500</real_time_update_rate><max_step_size>0.002</max_step_size></physics>')

# 场地地面
sdf.append('''    <model name="arena_floor">
      <static>true</static>
      <link name="link">
        <visual name="v">
          <pose>0 0 -0.005 0 0 0</pose>
          <geometry><box><size>20 20 0.01</size></box></geometry>
          <material><ambient>0.45 0.45 0.45 1</ambient></material>
        </visual>
      </link>
    </model>''')

idx = 0

# LINE -> box wall
for (x1,y1),(x2,y2) in lines:
    if not (in_track_area(x1,y1) or in_track_area(x2,y2)): continue
    dx, dy = x2-x1, y2-y1
    length = math.hypot(dx, dy)
    if length < MIN_LENGTH: continue
    mid_x = (x1+x2)/2 - cx
    mid_y = (y1+y2)/2 - cy
    angle = math.atan2(dy, dx)
    idx += 1
    sdf.append(f'''    <model name="wall_{idx}">
      <static>true</static>
      <link name="link">
        <collision name="c">
          <pose>{mid_x*scale:.4f} {mid_y*scale:.4f} {WALL_HEIGHT/2} 0 0 {angle:.4f}</pose>
          <geometry><box><size>{length*scale:.4f} {WALL_THICK} {WALL_HEIGHT}</size></box></geometry>
        </collision>
        <visual name="v">
          <pose>{mid_x*scale:.4f} {mid_y*scale:.4f} {WALL_HEIGHT/2} 0 0 {angle:.4f}</pose>
          <geometry><box><size>{length*scale:.4f} {WALL_THICK} {WALL_HEIGHT}</size></box></geometry>
          <material><ambient>0.2 0.2 0.7 1</ambient></material>
        </visual>
      </link>
    </model>''')

# ARC -> short box segments (过滤半径太小或不在赛道区域的圆弧)
for arc_cx, arc_cy, r, sa, ea in arcs:
    if r < MIN_LENGTH or not in_track_area(arc_cx, arc_cy): continue
    r = r * scale
    cx_a = (arc_cx - cx) * scale
    cy_a = (arc_cy - cy) * scale
    sa_r, ea_r = math.radians(sa), math.radians(ea)
    if ea_r < sa_r: ea_r += 2*math.pi
    num_seg = max(8, int((ea_r - sa_r) / (math.pi/12)))
    for i in range(num_seg):
        a1 = sa_r + (ea_r - sa_r) * i / num_seg
        a2 = sa_r + (ea_r - sa_r) * (i+1) / num_seg
        am = (a1 + a2) / 2
        x1 = cx_a + r * math.cos(a1)
        y1 = cy_a + r * math.sin(a1)
        x2 = cx_a + r * math.cos(a2)
        y2 = cy_a + r * math.sin(a2)
        len_seg = math.hypot(x2-x1, y2-y1)
        if len_seg < 0.01: continue
        mx = (x1+x2)/2
        my = (y1+y2)/2
        ang = math.atan2(y2-y1, x2-x1)
        idx += 1
        sdf.append(f'''    <model name="wall_arc_{idx}">
          <static>true</static>
          <link name="link">
            <collision name="c">
              <pose>{mx:.4f} {my:.4f} {WALL_HEIGHT/2} 0 0 {ang:.4f}</pose>
              <geometry><box><size>{len_seg:.4f} {WALL_THICK} {WALL_HEIGHT}</size></box></geometry>
            </collision>
            <visual name="v">
              <pose>{mx:.4f} {my:.4f} {WALL_HEIGHT/2} 0 0 {ang:.4f}</pose>
              <geometry><box><size>{len_seg:.4f} {WALL_THICK} {WALL_HEIGHT}</size></box></geometry>
              <material><ambient>0.2 0.2 0.7 1</ambient></material>
            </visual>
          </link>
        </model>''')

# CIRCLE -> cylinder obstacles (过滤太小或不在赛道区域的圆)
for circle_cx, circle_cy, r in circles:
    if r < MIN_LENGTH or not in_track_area(circle_cx, circle_cy): continue
    cx_c = (circle_cx - cx) * scale
    cy_c = (circle_cy - cy) * scale
    r = r * scale
    if r < 0.01: continue
    idx += 1
    sdf.append(f'''    <model name="pillar_{idx}">
      <static>true</static>
      <link name="link">
        <collision name="c">
          <pose>{cx_c*scale:.4f} {cy_c*scale:.4f} {WALL_HEIGHT/2} 0 0 0</pose>
          <geometry><cylinder><radius>{r:.4f}</radius><length>{WALL_HEIGHT}</length></cylinder></geometry>
        </collision>
        <visual name="v">
          <pose>{cx_c*scale:.4f} {cy_c*scale:.4f} {WALL_HEIGHT/2} 0 0 0</pose>
          <geometry><cylinder><radius>{r:.4f}</radius><length>{WALL_HEIGHT}</length></cylinder></geometry>
          <material><ambient>0.3 0.3 0.3 1</ambient></material>
        </visual>
      </link>
    </model>''')

# POLYLINE -> wall segments
for poly in polylines:
    # 跳过不在赛道区域的多段线
    if not any(in_track_area(v[0], v[1]) for v in poly): continue
    for i in range(len(poly)-1):
        x1, y1 = poly[i]
        x2, y2 = poly[i+1]
        dx, dy = x2-x1, y2-y1
        length = math.hypot(dx, dy)
        if length < MIN_LENGTH: continue
        mid_x = (x1+x2)/2 - cx
        mid_y = (y1+y2)/2 - cy
        angle = math.atan2(dy, dx)
        idx += 1
        sdf.append(f'''    <model name="polywall_{idx}">
      <static>true</static>
      <link name="link">
        <collision name="c">
          <pose>{mid_x*scale:.4f} {mid_y*scale:.4f} {WALL_HEIGHT/2} 0 0 {angle:.4f}</pose>
          <geometry><box><size>{length*scale:.4f} {WALL_THICK} {WALL_HEIGHT}</size></box></geometry>
        </collision>
        <visual name="v">
          <pose>{mid_x*scale:.4f} {mid_y*scale:.4f} {WALL_HEIGHT/2} 0 0 {angle:.4f}</pose>
          <geometry><box><size>{length*scale:.4f} {WALL_THICK} {WALL_HEIGHT}</size></box></geometry>
          <material><ambient>0.2 0.2 0.7 1</ambient></material>
        </visual>
      </link>
    </model>''')

sdf.append('  </world>')
sdf.append('</sdf>')

with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(sdf))

print(f"完成！共 {idx} 个墙体元素，已写入 {OUT_PATH}")
