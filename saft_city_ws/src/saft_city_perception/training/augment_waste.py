"""
增强垃圾桶数据并合并到现有训练集
"""
import json, sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from augment import (
    read_yolo_label, horizontal_flip, color_jitter,
    scale_with_pad, rotate_with_pad, clip_bboxes, load_image
)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
WASTE_DIR = DATA_DIR / "waste_data"
AUG_DIR = DATA_DIR / "augmented"

CLASS_NAMES = [
    "collapsed building", "building on fire",
    "building with toxic gas", "building with a power failure",
    "medical rescue groups", "general rescue groups",
    "kitchen waste bin", "recyclable waste bin",
    "hazardous waste bin", "residual waste bin",
]
NUM_CLASSES = len(CLASS_NAMES)
TARGET_SIZE = (640, 640)

per_img_augs = [
    ("orig",     lambda img, bboxes: (img.copy(), [list(b) for b in bboxes])),
    ("flip",     horizontal_flip),
    ("jbright",  lambda img, bboxes: color_jitter(img, bboxes, brightness=1.15)),
    ("jdark",    lambda img, bboxes: color_jitter(img, bboxes, brightness=0.88)),
    ("scale_09", lambda img, bboxes: scale_with_pad(img, bboxes, 0.90)),
    ("rot_p5",   lambda img, bboxes: rotate_with_pad(img, bboxes, 5)),
]

# 读取 waste_data 中的所有图片
img_paths = sorted(WASTE_DIR.glob("model/*.jpg"))
print(f"发现 {len(img_paths)} 张垃圾桶图片")

# 按 YOLO 格式读取标签
categories = [
    {"id": i + 1, "name": name, "supercategory": "object"}
    for i, name in enumerate(CLASS_NAMES)
]

aug_images = []
aug_annotations = []
img_id_counter = 50000
ann_id_counter = 50000

for img_path in img_paths:
    img = load_image(img_path)
    if img is None:
        print(f"  WARNING: 无法读取 {img_path}")
        continue

    h, w = img.shape[:2]
    label_path = WASTE_DIR / "labels" / f"{img_path.stem}.txt"
    bboxes_px = []
    class_ids = []
    if label_path.exists():
        bboxes_px, class_ids_yolo = read_yolo_label(label_path, w, h)
        class_ids = [cid + 1 for cid in class_ids_yolo]  # YOLO 0-based → COCO 1-based

    if not bboxes_px:
        print(f"  SKIP: {img_path.name} 无标签")
        continue

    for aug_name, aug_fn in per_img_augs:
        aug_img, aug_bboxes = aug_fn(img, bboxes_px)
        aug_bboxes = clip_bboxes(aug_bboxes, aug_img.shape[1], aug_img.shape[0])
        if not aug_bboxes:
            continue

        aug_img_resized = cv2.resize(aug_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        sx, sy = 640.0 / aug_img.shape[1], 640.0 / aug_img.shape[0]

        out_name = f"{img_path.stem}_{aug_name}.jpg"
        cv2.imwrite(str(AUG_DIR / "image" / out_name), aug_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        new_img_id = img_id_counter
        img_id_counter += 1
        aug_images.append({"id": new_img_id, "file_name": out_name, "width": 640, "height": 640})

        for bbox, cid in zip(aug_bboxes, class_ids):
            sb = [round(bbox[0] * sx, 2), round(bbox[1] * sy, 2),
                  round(bbox[2] * sx, 2), round(bbox[3] * sy, 2)]
            aug_annotations.append({
                "id": ann_id_counter, "image_id": new_img_id,
                "category_id": cid, "bbox": sb,
                "area": round(sb[2] * sb[3], 2), "iscrowd": 0,
            })
            ann_id_counter += 1

print(f"新增: {len(aug_images)} 张图, {len(aug_annotations)} 个标注")

# 合并到现有 train.json
train_path = AUG_DIR / "train.json"
with open(train_path) as f:
    train = json.load(f)

old_count = len(train["images"])
old_ann_count = len(train["annotations"])
train["images"].extend(aug_images)
train["annotations"].extend(aug_annotations)

with open(train_path, "w") as f:
    json.dump(train, f, indent=2)

print(f"合并完成: {old_count} → {len(train['images'])} 张图, {old_ann_count} → {len(train['annotations'])} 个标注")

# 统计垃圾桶类
from collections import Counter
cat_cnt = Counter()
for ann in train["annotations"]:
    if ann["category_id"] in (7, 8, 9, 10):
        cat_cnt[ann["category_id"]] += 1
cat_names = {c["id"]: c["name"] for c in train["categories"]}
print("\n垃圾桶类分布:")
for cid in sorted(cat_cnt):
    print(f"  {cat_names[cid]}: {cat_cnt[cid]}")
