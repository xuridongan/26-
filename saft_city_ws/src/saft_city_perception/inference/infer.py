"""
PP-YOLOE+ S 安全城市场景推理脚本（支持滑窗推理）

用法:
  # 常规推理
  python infer.py --image_dir data/image

  # 滑窗推理（大拼接图）
  python infer.py --image_dir D:/saft_city/test --sliding_window \\
      --window_width 3072 --window_height 4096 --stride 2000 --threshold 0.25 \\
      --output D:/saft_city/test/sliding_results
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import paddle
from PIL import Image, ImageDraw, ImageFont

from ppdet.core.workspace import create, load_config
from ppdet.utils.checkpoint import load_weight
from ppdet.utils.logger import setup_logger

logger = setup_logger(__name__)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

CLASS_NAMES = [
    "倒塌建筑", "着火建筑",
    "毒气建筑", "电力故障建筑",
    "医疗救援组", "通用救援组",
    "厨余垃圾", "可回收垃圾",
    "有害垃圾", "其他垃圾",
]
NUM_CLASSES = len(CLASS_NAMES)

# 为每类分配颜色 (BGR)
COLORS = [
    (0, 0, 255),    # red - collapsed building
    (255, 0, 0),    # blue - building on fire
    (0, 255, 255),  # yellow - building with toxic gas
    (255, 0, 255),  # magenta - building with power failure
    (0, 255, 0),    # green - medical rescue
    (255, 255, 0),  # cyan - general rescue
    (128, 0, 128),  # purple - kitchen waste
    (255, 165, 0),  # orange - recyclable waste
    (128, 128, 0),  # olive - hazardous waste
    (0, 128, 128),  # teal - residual waste
]


def write_infer_config(score_thresh, weights_path):
    """写推理用的 config."""
    cfg = {
        "architecture": "PPYOLOE",
        "weights": str(weights_path),
        "num_classes": len(CLASS_NAMES),
        "metric": "COCO",
        "backbone": "CSPResNet",
        "CSPResNet": {
            "act": "swish",
            "depth_mult": 0.33, "width_mult": 0.50,
            "return_idx": [1, 2, 3],
            "use_large_stem": True,
        },
        "neck": "CustomCSPPAN",
        "CustomCSPPAN": {
            "act": "swish",
            "depth_mult": 0.33, "width_mult": 0.50,
        },
        "yolo_head": "PPYOLOEHead",
        "PPYOLOEHead": {
            "num_classes": len(CLASS_NAMES),
            "act": "swish",
            "fpn_strides": [32, 16, 8],
            "grid_cell_scale": 5.0, "grid_cell_offset": 0.5, "reg_max": 16,
            "static_assigner_epoch": 4,
            "nms": "MultiClassNMS",
        },
        "MultiClassNMS": {
            "score_threshold": score_thresh, "nms_top_k": 1000,
            "keep_top_k": 100, "nms_threshold": 0.5,
        },
        "post_process": "BBoxPostProcess",
        "BBoxPostProcess": {
            "decode": "PPYOLOEDecode",
            "nms": "MultiClassNMS",
        },
        "PPYOLOEDecode": {"reg_max": 16},
        "worker_num": 1,
        "use_gpu": paddle.is_compiled_with_cuda(),
    }

    # Transform for single image inference
    cfg["EvalReader"] = {
        "sample_transforms": [
            {"Decode": {}},
            {"Resize": {"target_size": [640, 640], "keep_ratio": False, "interp": 2}},
            {"NormalizeImage": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "is_scale": True, "norm_type": "mean_std",
            }},
            {"Permute": {}},
        ],
        "batch_size": 1,
    }

    import yaml as _yaml
    yml_path = PROJECT_DIR / "_infer_config.yaml"
    with open(yml_path, "w") as f:
        _yaml.dump(cfg, f, default_flow_style=None, width=120)
    return yml_path


def build_infer_model(score_thresh, weights_path):
    """Build model and loader for inference."""
    yml_path = write_infer_config(score_thresh, weights_path)
    cfg = load_config(str(yml_path))
    model = create(cfg.architecture)
    model.post_process = None  # 使用 head 自带的 post_process
    load_weight(model, cfg.weights)
    model.eval()
    return model


def preprocess_image(img_path):
    """加载并预处理单张图片，返回模型输入 dict."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return _preprocess_array(img)


def _preprocess_array(img_bgr):
    """对 BGR numpy 数组做预处理，返回模型输入 dict."""
    from ppdet.data.transform.operators import Resize, NormalizeImage, Permute

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    sample = {"image": img, "h": orig_h, "w": orig_w}
    sample = Resize(target_size=[640, 640], keep_ratio=False, interp=2)(sample)
    sample = NormalizeImage(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True, norm_type="mean_std",
    )(sample)
    sample = Permute()(sample)

    if sample is None:
        raise RuntimeError("Transform failed on image array")

    im_shape = np.array([orig_h, orig_w], dtype=np.float32).reshape(1, 2)
    scale_factor = np.array([1.0, 1.0], dtype=np.float32).reshape(1, 2)

    data = {
        "image": paddle.to_tensor(sample["image"]).unsqueeze(0),
        "im_shape": paddle.to_tensor(im_shape),
        "scale_factor": paddle.to_tensor(scale_factor),
    }
    return data, (orig_h, orig_w)


def _run_model(model, data, orig_size, score_thresh, class_thresholds=None):
    """运行模型推理并将 640x640 坐标映射回原图。

    当 class_thresholds 指定时，score_thresh 用作内部低阈值 (推荐 0.01)，
    返回后再按逐类阈值过滤。
    """
    h, w = orig_size

    # 使用全局阈值（当有逐类阈值时，传低阈值 0.01 进来）
    old_thresh = model.yolo_head.nms.score_threshold
    model.yolo_head.nms.score_threshold = score_thresh

    with paddle.no_grad():
        outs = model(data)

    model.yolo_head.nms.score_threshold = old_thresh

    bbox = outs["bbox"].numpy()       # [N, 6] = [class_id, score, x1, y1, x2, y2]
    bbox_num = outs["bbox_num"].numpy()

    detections = []
    scale_x = w / 640.0
    scale_y = h / 640.0

    num_dets = int(bbox_num.flatten()[0])
    for i in range(num_dets):
        det = bbox[i]
        cls_id = int(det[0])
        score = float(det[1])
        x1, y1, x2, y2 = det[2:6].tolist()

        # 按逐类阈值过滤
        if class_thresholds is not None:
            if 0 <= cls_id < len(class_thresholds) and score < class_thresholds[cls_id]:
                continue

        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y

        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
            "score": score,
            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
            "bbox_pixel": [int(x1), int(y1), int(x2), int(y2)],
        })

    return detections


def infer_image(model, img_path, score_thresh, class_thresholds=None):
    """对单张图片推理，返回检测结果。"""
    data, orig_size = preprocess_image(img_path)
    return _run_model(model, data, orig_size, score_thresh, class_thresholds)


def infer_image_array(model, img_bgr, score_thresh, class_thresholds=None):
    """对 BGR numpy 数组推理，返回检测结果。"""
    data, orig_size = _preprocess_array(img_bgr)
    return _run_model(model, data, orig_size, score_thresh, class_thresholds)


def bbox_iou(box1, box2):
    """Compute IoU of two bounding boxes [x1, y1, x2, y2]."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def nms_merge(detections, nms_threshold=0.5):
    """按类 NMS 合并滑窗产生的重叠检测框。"""
    if not detections:
        return []

    by_class = {}
    for det in detections:
        by_class.setdefault(det["class_id"], []).append(det)

    merged = []
    for cls_id in sorted(by_class.keys()):
        cls_dets = sorted(by_class[cls_id], key=lambda x: x["score"], reverse=True)
        keep = []
        while cls_dets:
            best = cls_dets.pop(0)
            keep.append(best)
            b1 = best["bbox_pixel"]
            cls_dets = [
                d for d in cls_dets
                if bbox_iou(b1, d["bbox_pixel"]) <= nms_threshold
            ]
        merged.extend(keep)

    return merged


def cross_class_nms(detections, iou_threshold=0.5):
    """跨类 NMS: 不同类别的框若重叠超过阈值，只保留分数最高的一个。"""
    if not detections:
        return []
    dets = sorted(detections, key=lambda x: x["score"], reverse=True)
    keep = []
    for d in dets:
        b1 = d["bbox_pixel"]
        # 与已保留的所有框比较 IoU
        suppressed = False
        for k in keep:
            b2 = k["bbox_pixel"]
            if bbox_iou(b1, b2) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
    return keep


def filter_by_class_thresholds(detections, class_thresholds):
    """按逐类阈值过滤检测结果。class_thresholds 是 [th0, th1, ..., th9]。"""
    if class_thresholds is None:
        return detections
    filtered = []
    for d in detections:
        cls_id = d["class_id"]
        if 0 <= cls_id < len(class_thresholds):
            if d["score"] >= class_thresholds[cls_id]:
                filtered.append(d)
        else:
            filtered.append(d)  # 未知类不过滤
    return filtered


def sliding_window_infer(model, img_bgr, score_thresh,
                         window_size, stride, merge_nms=0.5,
                         class_thresholds=None):
    """
    滑窗推理大图。

    Args:
        model: 推理模型
        img_bgr: 全图 BGR numpy 数组
        score_thresh: 置信度阈值（有逐类阈值时用 0.01）
        window_size: (w, h) 窗口尺寸
        stride: (x_step, y_step) 步长
        merge_nms: 合并 NMS 的 IoU 阈值
        class_thresholds: 逐类阈值列表或 None

    Returns:
        list[dict]: 合并后的检测结果
    """
    h, w = img_bgr.shape[:2]
    win_w, win_h = window_size
    stride_x, stride_y = stride

    all_detections = []
    total_windows = 0

    # 如果图片某边小于窗口尺寸，缩小窗口到图片尺寸
    win_w = min(win_w, w)
    win_h = min(win_h, h)

    # 生成所有窗口位置（从左上到右下，确保覆盖边缘）
    positions = []
    for y in range(0, h - win_h + 1, stride_y):
        for x in range(0, w - win_w + 1, stride_x):
            positions.append((x, y, x + win_w, y + win_h))

    # 确保右下角被覆盖
    if positions:
        last_x = w - win_w
        last_y = h - win_h
        if (last_x, last_y) not in {(x, y) for x, y, _, _ in positions}:
            positions.append((last_x, last_y, last_x + win_w, last_y + win_h))

    # 去重（边缘对齐后可能重复）
    positions = list(set(positions))

    logger.info(f"  滑窗: {len(positions)} 个窗口 ({win_w}x{win_h}, 步长 {stride_x}x{stride_y})")

    for idx, (x1, y1, x2, y2) in enumerate(positions):
        crop = img_bgr[y1:y2, x1:x2]
        dets = infer_image_array(model, crop, score_thresh, class_thresholds)

        # 将坐标偏移回原图
        for d in dets:
            d["bbox_pixel"][0] += x1
            d["bbox_pixel"][1] += y1
            d["bbox_pixel"][2] += x1
            d["bbox_pixel"][3] += y1
            # 更新 bbox (x, y, w, h) 也偏移
            d["bbox"][0] = round(d["bbox_pixel"][0], 2)
            d["bbox"][1] = round(d["bbox_pixel"][1], 2)

        all_detections.extend(dets)

        if (idx + 1) % 10 == 0 or idx == len(positions) - 1:
            logger.info(f"    窗口 [{idx+1}/{len(positions)}] 当前累积 {len(all_detections)} 个检测")

    # 全局 NMS 合并（同类别）
    before = len(all_detections)
    merged = nms_merge(all_detections, merge_nms)
    logger.info(f"  NMS 合并: {before} → {len(merged)} 个检测")

    # 跨类 NMS：不同类但高度重叠只保留分数最高的
    before = len(merged)
    merged = cross_class_nms(merged, iou_threshold=0.5)
    if len(merged) < before:
        logger.info(f"  跨类 NMS: {before} → {len(merged)} 个检测")

    return merged


def draw_detections(img, detections, orig_size):
    """在图片上绘制检测框和标签。"""
    h, w = orig_size
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox_pixel"]
        cls_id = det["class_id"]
        score = det["score"]
        cls_name = det["class_name"]
        color = COLORS[cls_id % len(COLORS)]

        draw.rectangle([x1, y1, x2, y2], outline=tuple(reversed(color)), width=3)

        label = f"{cls_name} {score:.2f}"
        label_bbox = draw.textbbox((0, 0), label, font=font)
        tw = label_bbox[2] - label_bbox[0]
        th = label_bbox[3] - label_bbox[1]
        draw.rectangle([x1, y1 - th - 8, x1 + tw + 8, y1], fill=tuple(reversed(color)))
        draw.text((x1 + 4, y1 - th - 4), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="PP-YOLOE+ S 推理")
    parser.add_argument("--weights", type=str,
                        default=str(OUTPUT_DIR / "model_final.pdparams"))
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="置信度阈值 (默认 0.25)")
    parser.add_argument("--image_path", type=str, default=None,
                        help="单张图片路径")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="图片目录")
    parser.add_argument("--output", type=str, default=str(PROJECT_DIR / "infer_results"),
                        help="输出目录")
    parser.add_argument("--save_json", action="store_true", default=True,
                        help="保存 JSON 结果")
    parser.add_argument("--save_image", action="store_true", default=True,
                        help="保存可视化图片")

    # 逐类阈值
    parser.add_argument("--class_thresholds", type=str, default=None,
                        help="逐类阈值，逗号分隔的10个值，如 '0.35,0.20,0.15,...'。"
                             "不指定则用 --threshold 统一阈值。")
    parser.add_argument("--auto_class_thresholds", action="store_true",
                        help="自动从 class_thresholds.json 读取逐类阈值")

    # 滑窗参数
    parser.add_argument("--sliding_window", action="store_true",
                        help="启用滑窗推理（用于大拼接图）")
    parser.add_argument("--window_width", type=int, default=3072,
                        help="滑窗宽度 (默认 3072)")
    parser.add_argument("--window_height", type=int, default=4096,
                        help="滑窗高度 (默认 4096)")
    parser.add_argument("--stride", type=int, default=2000,
                        help="滑窗步长 (默认 2000，约 50-65% 重叠)")
    parser.add_argument("--merge_nms", type=float, default=0.3,
                        help="合并 NMS 的 IoU 阈值 (默认 0.3)")

    args = parser.parse_args()

    # 解析逐类阈值
    class_thresholds = None
    if args.auto_class_thresholds:
        ct_path = PROJECT_DIR / "class_thresholds.json"
        if ct_path.exists():
            with open(ct_path) as f:
                ct_data = json.load(f)
            if "thresholds_list" in ct_data:
                class_thresholds = ct_data["thresholds_list"]
            else:
                # 格式: {"1": 0.15, "2": 0.05, ...}
                class_thresholds = [float(ct_data.get(str(i+1), 0.25)) for i in range(NUM_CLASSES)]
            logger.info(f"自动加载逐类阈值: {class_thresholds}")
        else:
            logger.warning(f"class_thresholds.json 不存在: {ct_path}")
    elif args.class_thresholds:
        parts = [x.strip() for x in args.class_thresholds.split(",")]
        if len(parts) != NUM_CLASSES:
            parser.error(f"--class_thresholds 需要 {NUM_CLASSES} 个值，收到 {len(parts)}")
        class_thresholds = [float(p) for p in parts]
        logger.info(f"逐类阈值: {class_thresholds}")

    # 参数校验
    if not args.image_path and not args.image_dir:
        parser.error("请指定 --image_path 或 --image_dir")
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"权重文件不存在: {weights_path}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片
    if args.image_path:
        img_paths = [Path(args.image_path)]
    else:
        img_dir = Path(args.image_dir)
        img_paths = sorted([
            p for p in img_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ])

    if not img_paths:
        logger.error("未找到图片")
        return

    logger.info(f"模型: {weights_path}")
    logger.info(f"阈值: {args.threshold}")
    if class_thresholds:
        logger.info(f"逐类阈值: {class_thresholds}")
    logger.info(f"图片数: {len(img_paths)}")
    if args.sliding_window:
        logger.info(f"滑窗: {args.window_width}x{args.window_height}, 步长 {args.stride}")
    logger.info("=" * 50)

    # 有逐类阈值时，内部模型用低阈值 0.01 推理，再按类过滤
    infer_thresh = 0.01 if class_thresholds else args.threshold
    model = build_infer_model(infer_thresh, weights_path)

    total_time = 0.0
    all_results = {}

    for img_path in img_paths:
        logger.info(f"推理: {img_path.name} ...")

        import time
        t_start = time.time()

        if args.sliding_window and args.sliding_window:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.error(f"  无法读取: {img_path}")
                continue
            detections = sliding_window_infer(
                model, img_bgr, infer_thresh,
                window_size=(args.window_width, args.window_height),
                stride=(args.stride, args.stride),
                merge_nms=args.merge_nms,
                class_thresholds=class_thresholds,
            )
            orig_h, orig_w = img_bgr.shape[:2]
        else:
            detections = infer_image(model, img_path, infer_thresh, class_thresholds)
            detections = cross_class_nms(detections, iou_threshold=0.5)
            img_check = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            orig_h, orig_w = img_check.shape[:2]

        elapsed = time.time() - t_start
        total_time += elapsed

        logger.info(f"  检测到 {len(detections)} 个目标 ({elapsed:.2f}s)")

        # 保存 JSON
        if args.save_json:
            json_data = {
                "image": img_path.name,
                "image_width": orig_w,
                "image_height": orig_h,
                "threshold": args.threshold,
                "sliding_window": args.sliding_window,
                "detections": [
                    {
                        "class_id": d["class_id"],
                        "class_name": d["class_name"],
                        "score": d["score"],
                        "bbox": d["bbox"],
                    }
                    for d in detections
                ],
            }
            json_path = out_dir / f"{img_path.stem}.json"
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
            all_results[img_path.name] = json_data["detections"]

        # 保存可视化
        if args.save_image and detections:
            if args.sliding_window:
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            else:
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            vis = draw_detections(img_bgr, detections, (orig_h, orig_w))
            vis_path = out_dir / f"{img_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis)
            logger.info(f"  可视化: {vis_path}")

    if img_paths:
        avg_time = total_time / len(img_paths)
        logger.info(f"\n平均推理时间: {avg_time:.2f}s/张 ({1.0/avg_time:.1f} FPS)")

    if args.save_json:
        all_path = out_dir / "all_results.json"
        with open(all_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"全部结果: {all_path}")

    logger.info(f"完成! 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
