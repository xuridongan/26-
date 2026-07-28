"""
对 D:\saft_city\result 中的图片进行推理并标注，与虚拟机推理流程一致
"""
import cv2, os, sys, json, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'inference'))
from infer import build_infer_model, infer_image_array

# 加载模型
model = build_infer_model(0.01, os.path.join(BASE_DIR, 'inference', 'model_final.pdparams'))

# 分类偏置调整
MEDICAL_IDX, GENERAL_IDX = 4, 5
for i in range(len(model.yolo_head.pred_cls)):
    bias = model.yolo_head.pred_cls[i].bias
    bias_np = bias.numpy()
    bias_np[MEDICAL_IDX] += 1.0
    bias_np[GENERAL_IDX] -= 1.0
    bias.set_value(bias_np)

# 阈值
with open(os.path.join(BASE_DIR, 'inference', 'class_thresholds.json')) as f:
    raw = json.load(f)
thresholds = [raw.get(str(i+1), 0.05) for i in range(10)]

CLASS_NAMES = [
    "倒塌建筑","着火建筑","毒气建筑",
    "电力故障建筑","医疗救援组","通用救援组",
    "厨余垃圾","可回收垃圾","有害垃圾","其他垃圾",
]
COLORS = [(0,0,255),(255,0,0),(0,255,255),(255,0,255),(0,255,0),(255,255,0),(128,0,128),(255,165,0),(128,128,0),(0,128,128)]

# 字体
try:
    font = ImageFont.truetype("simhei.ttf", 20)
except:
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
    except:
        font = ImageFont.load_default()

SRC_DIR = r"D:\saft_city\result"
OUT_DIR = os.path.join(SRC_DIR, "annotated")
os.makedirs(OUT_DIR, exist_ok=True)

img_paths = sorted([
    os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"找到 {len(img_paths)} 张图片\n")

for img_path in img_paths:
    name = os.path.basename(img_path)
    print(f"处理: {name} ...", end=" ")

    img = cv2.imread(img_path)
    if img is None:
        print("无法读取")
        continue

    t0 = time.time()
    dets = infer_image_array(model, img, 0.01, thresholds)
    infer_ms = (time.time() - t0) * 1000

    # 优先级 NMS: 医疗救援(4)和通用救援(5)重叠时保留医疗
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
                keep[i] = False
                break
    dets = [d for i, d in enumerate(dets) if keep[i]]

    # 画框
    for d in dets:
        x1, y1, x2, y2 = d["bbox_pixel"]
        color = COLORS[d["class_id"] % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 中文标签
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
    cv2.putText(img, f"Detected: {len(dets)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    out_path = os.path.join(OUT_DIR, name)
    cv2.imwrite(out_path, img)
    print(f"{len(dets)} 个检测 ({infer_ms:.0f}ms) → {out_path}")

print(f"\n完成！标注图保存在 {OUT_DIR}")
