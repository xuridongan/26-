"""
离线增强脚本（多目标版）
读取 data/for_training/ 的 YOLO 格式数据，增强后输出到 data/augmented/

策略:
  - identity / flip / 温和色彩抖动 / 缩放保留全图 / 旋转 / 模糊 / 锐化
  - 垃圾桶类额外增强
  - 其他垃圾类额外增强
"""
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

random.seed(42)
np.random.seed(42)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
SRC_IMAGE_DIR = DATA_DIR / "data" / "model"
SRC_LABEL_DIR = DATA_DIR / "data" / "labels_my-project-name_2026-07-05-07-40-02"

VAL_IMAGE_DIR = DATA_DIR / "image"  # 原图复制到这里供验证集使用
AUG_DIR = DATA_DIR / "augmented"
AUG_IMAGE_DIR = AUG_DIR / "image"

CLASS_NAMES = [
    "collapsed building", "building on fire",
    "building with toxic gas", "building with a power failure",
    "medical rescue groups", "general rescue groups",
    "kitchen waste bin", "recyclable waste bin",
    "hazardous waste bin", "residual waste bin",
]

# ---------- YOLO 格式读取 ----------

def read_yolo_label(label_path, img_w, img_h):
    """读取 YOLO 标签，返回 list of [x, y, w, h] (像素坐标) 和对应的 class_id 列表。"""
    bboxes = []
    class_ids = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x = (cx - bw / 2) * img_w
                y = (cy - bh / 2) * img_h
                bw = bw * img_w
                bh = bh * img_h
                bboxes.append([x, y, bw, bh])
                class_ids.append(cls_id)
    return bboxes, class_ids


# ---------- 增强函数（多 bbox 版本）----------

def horizontal_flip(img, bboxes):
    h, w = img.shape[:2]
    img = cv2.flip(img, 1)
    return img, [[w - x - bw, y, bw, bh] for x, y, bw, bh in bboxes]


def color_jitter(img, bboxes, brightness=1.0, contrast=1.0, saturation=1.0):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if brightness != 1.0:
        pil = ImageEnhance.Brightness(pil).enhance(brightness)
    if contrast != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(contrast)
    if saturation != 1.0:
        pil = ImageEnhance.Color(pil).enhance(saturation)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), [list(b) for b in bboxes]


def scale_with_pad(img, bboxes, scale):
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x_off = (w - new_w) // 2
    y_off = (h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    new_bboxes = []
    for bbox in bboxes:
        x, y, bw, bh = bbox
        new_bboxes.append([x * scale + x_off, y * scale + y_off, bw * scale, bh * scale])
    return canvas, new_bboxes


def rotate_with_pad(img, bboxes, angle):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    new_bboxes = []
    for bbox in bboxes:
        x, y, bw, bh = bbox
        corners = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
                           dtype=np.float32)
        corners_t = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
        nx = min(corners_t[:, 0])
        ny = min(corners_t[:, 1])
        new_bboxes.append([nx, ny, max(corners_t[:, 0]) - nx, max(corners_t[:, 1]) - ny])

    return rotated, new_bboxes


def gaussian_blur(img, bboxes, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0), [list(b) for b in bboxes]


def sharpness_enhance(img, bboxes, factor=2.0):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Sharpness(pil).enhance(factor)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), [list(b) for b in bboxes]


def load_image(img_path):
    return cv2.imread(str(img_path), cv2.IMREAD_COLOR)


# ---------- 工具函数 ----------

def clip_bboxes(bboxes, img_w, img_h):
    """将 bbox 裁剪到图像范围内，过滤掉无效的。"""
    valid = []
    for b in bboxes:
        x = max(0, min(b[0], img_w - 1))
        y = max(0, min(b[1], img_h - 1))
        bw = max(1, min(b[2], img_w - x))
        bh = max(1, min(b[3], img_h - y))
        valid.append([x, y, bw, bh])
    return valid


# ---------- 主流程 ----------

def main():
    SRC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SRC_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    VAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    AUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(SRC_IMAGE_DIR.glob("*.jpg"))
    if not img_paths:
        print("错误: 未找到图片")
        return

    print(f"发现 {len(img_paths)} 张源图片")

    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(CLASS_NAMES)
    ]

    # ========== 1. 验证集：原图 ==========
    val_images = []
    val_annotations = []
    ann_id = 1

    for img_id, img_path in enumerate(img_paths, start=1):
        shutil.copy2(str(img_path), str(VAL_IMAGE_DIR / img_path.name))
        with Image.open(img_path) as pimg:
            w, h = pimg.size
        val_images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})

        label_path = SRC_LABEL_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            bboxes_px, class_ids = read_yolo_label(label_path, w, h)
            for bbox, cls_id in zip(bboxes_px, class_ids):
                val_annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cls_id + 1,
                    "bbox": [round(v, 2) for v in bbox],
                    "area": round(bbox[2] * bbox[3], 2), "iscrowd": 0,
                })
                ann_id += 1

    with open(AUG_DIR / "val.json", "w") as f:
        json.dump({"images": val_images, "annotations": val_annotations, "categories": categories}, f, indent=2)
    print(f"验证集: {len(val_images)} 张图, {len(val_annotations)} 个标注")

    # ========== 2. 训练集增强 ==========
    aug_images = []
    aug_annotations = []
    img_id_counter = 1000
    ann_id_counter = 10000

    TARGET_SIZE = (640, 640)

    # --- 每张图 6 种增强（~1000 张）---
    per_img_augs = [
        ("orig",        lambda img, bboxes: (img.copy(), [list(b) for b in bboxes])),
        ("flip",        horizontal_flip),
        ("jbright",     lambda img, bboxes: color_jitter(img, bboxes, brightness=1.15)),
        ("jdark",       lambda img, bboxes: color_jitter(img, bboxes, brightness=0.88)),
        ("scale_09",    lambda img, bboxes: scale_with_pad(img, bboxes, 0.90)),
        ("rot_p5",      lambda img, bboxes: rotate_with_pad(img, bboxes, 5)),
    ]

    for img_info in val_images:
        img_path = SRC_IMAGE_DIR / img_info["file_name"]
        img = load_image(img_path)
        if img is None:
            print(f"  WARNING: 无法读取 {img_path}")
            continue

        orig_h, orig_w = img.shape[:2]

        # 收集此图所有 bbox
        all_bboxes = []
        all_cat_ids = []
        for ann in val_annotations:
            if ann["image_id"] == img_info["id"]:
                all_bboxes.append(list(ann["bbox"]))
                all_cat_ids.append(ann["category_id"])

        has_bboxes = bool(all_bboxes)

        for aug_name, aug_fn in per_img_augs:
            if has_bboxes:
                aug_img, aug_bboxes = aug_fn(img, all_bboxes)
                aug_bboxes = clip_bboxes(aug_bboxes, aug_img.shape[1], aug_img.shape[0])
                if not aug_bboxes:
                    continue
            else:
                # 空背景：只 resize，无 bbox 处理
                aug_img = img.copy()
                aug_bboxes = []
                all_cat_ids = []

            aug_img_resized = cv2.resize(aug_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            sx, sy = 640.0 / aug_img.shape[1], 640.0 / aug_img.shape[0]

            stem = Path(img_info["file_name"]).stem
            out_name = f"{stem}_{aug_name}.jpg"
            cv2.imwrite(str(AUG_IMAGE_DIR / out_name), aug_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            new_img_id = img_id_counter
            img_id_counter += 1
            aug_images.append({"id": new_img_id, "file_name": out_name, "width": 640, "height": 640})

            for bbox, cid in zip(aug_bboxes, all_cat_ids):
                sb = [round(bbox[0] * sx, 2), round(bbox[1] * sy, 2),
                      round(bbox[2] * sx, 2), round(bbox[3] * sy, 2)]
                aug_annotations.append({
                    "id": ann_id_counter, "image_id": new_img_id,
                    "category_id": cid, "bbox": sb,
                    "area": round(sb[2] * sb[3], 2), "iscrowd": 0,
                })
                ann_id_counter += 1

    # ========== 保存训练集 JSON ==========
    train_coco = {
        "images": aug_images,
        "annotations": aug_annotations,
        "categories": categories,
    }
    with open(AUG_DIR / "train.json", "w") as f:
        json.dump(train_coco, f, indent=2)

    # 统计
    cat_cnt = Counter()
    for ann in aug_annotations:
        cat_cnt[ann["category_id"]] += 1
    cat_names = {c["id"]: c["name"] for c in categories}

    print(f"训练集: {len(aug_images)} 张图, {len(aug_annotations)} 个标注")
    print("按类别分布:")
    for cid in sorted(cat_cnt):
        print(f"  {cat_names[cid]}: {cat_cnt[cid]}")
    print(f"  → {AUG_IMAGE_DIR}")
    print(f"  → {AUG_DIR / 'train.json'}")
    print(f"  → {AUG_DIR / 'val.json'}")

    # 写 label_list.txt
    with open(DATA_DIR / "label_list.txt", "w") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")


if __name__ == "__main__":
    main()
