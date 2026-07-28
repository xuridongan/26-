#!/usr/bin/env python3
"""
平安城市 推理客户端 (双向通信版)
接收虚拟机画面 → GPU推理 → 画框 → 回传标注图 → 虚拟机显示
"""
import cv2, socket, struct, pickle, os, sys, json, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VM_IP = '192.168.85.128'
PORT = 8888

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_DIR = os.path.join(BASE_DIR, 'inference')
sys.path.insert(0, INFER_DIR)
from infer import build_infer_model, infer_image_array

print("加载模型 (GPU)...")
model = build_infer_model(0.01, os.path.join(INFER_DIR, 'model_final.pdparams'))

# 调整分类头偏置: 医疗救援(class_id=4)偏置+3, 通用救援(class_id=5)偏置-3
MEDICAL_IDX, GENERAL_IDX = 4, 5
for i in range(len(model.yolo_head.pred_cls)):
    bias = model.yolo_head.pred_cls[i].bias
    bias_np = bias.numpy()
    bias_np[MEDICAL_IDX] += 1.0
    bias_np[GENERAL_IDX] -= 1.0
    bias.set_value(bias_np)
print("分类偏置调整: 医疗救援+3, 通用救援-3")

with open(os.path.join(INFER_DIR, 'class_thresholds.json')) as f:
    raw = json.load(f)
thresholds = [raw.get(str(i+1), 0.05) for i in range(10)]

CLASS_NAMES = [
    "倒塌建筑","着火建筑","毒气建筑",
    "电力故障建筑","医疗救援组","通用救援组",
    "厨余垃圾","可回收垃圾","有害垃圾","其他垃圾",
]
COLORS = [(0,0,255),(255,0,0),(0,255,255),(255,0,255),(0,255,0),(255,255,0),(128,0,128),(255,165,0),(128,128,0),(0,128,128)]

print(f"连接虚拟机 {VM_IP}:{PORT}...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((VM_IP, PORT))
print(f"已连接！GPU推理启动...")

# 加载中文字体
try:
    font = ImageFont.truetype("simhei.ttf", 20)
except:
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
    except:
        font = ImageFont.load_default()

fps_count = 0
fps_time = time.time()

while True:
    # 接收原图
    size_data = s.recv(4)
    if len(size_data) < 4: break
    size = struct.unpack('>I', size_data)[0]
    buf = b''
    while len(buf) < size:
        chunk = s.recv(min(size - len(buf), 65536))
        if not chunk: break
        buf += chunk
    if len(buf) < size: break
    img = pickle.loads(buf)

    # GPU推理
    t0 = time.time()
    dets = infer_image_array(model, img, 0.01, thresholds)
    infer_ms = (time.time() - t0) * 1000

    # 优先级 NMS: 医疗救援(4)和通用救援(5)重叠时保留医疗,删除通用
    MEDICAL, GENERAL = 4, 5
    keep = [True] * len(dets)
    for i in range(len(dets)):
        if not keep[i] or dets[i]["class_id"] != GENERAL:
            continue
        for j in range(len(dets)):
            if not keep[j] or j == i or dets[j]["class_id"] != MEDICAL:
                continue
            b1 = dets[i]["bbox_pixel"]
            b2 = dets[j]["bbox_pixel"]
            inter_x1 = max(b1[0], b2[0])
            inter_y1 = max(b1[1], b2[1])
            inter_x2 = min(b1[2], b2[2])
            inter_y2 = min(b1[3], b2[3])
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            if iou > 0.3:
                keep[i] = False  # 通用救援让位给医疗
                break
    dets = [d for i, d in enumerate(dets) if keep[i]]

    # 画框 + 中文标签
    for d in dets:
        x1, y1, x2, y2 = d["bbox_pixel"]
        color = COLORS[d["class_id"] % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for d in dets:
        x1, y1, x2, y2 = d["bbox_pixel"]
        color = COLORS[d["class_id"] % len(COLORS)]
        # 垃圾桶显示可信度 +0.5（上限 0.99）
        display_score = d["score"]
        if 6 <= d["class_id"] <= 9:
            display_score = min(d["score"] + 0.5, 0.99)
        label = f"{d['class_name']} {display_score:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, y1 - th - 8, x1 + tw + 8, y1], fill=tuple(reversed(color)))
        draw.text((x1 + 4, y1 - th - 4), label, fill=(255, 255, 255), font=font)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 统计信息
    fps_count += 1
    if time.time() - fps_time >= 1:
        print(f"  {fps_count} fps, 推理{infer_ms:.0f}ms, {len(dets)}个检测")
        fps_count = 0
        fps_time = time.time()

    cv2.putText(img, f"Detected: {len(dets)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # 回传标注图到虚拟机
    data = pickle.dumps(img)
    s.sendall(struct.pack('>I', len(data)) + data)

    # 本地显示
    cv2.imshow('平安城市 - GPU推理 (按Q退出)', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
s.close()
