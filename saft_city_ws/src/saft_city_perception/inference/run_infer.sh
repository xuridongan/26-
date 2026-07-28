#!/bin/bash
# 推理脚本 - 安全城市场景检测
# 用法:
#   ./run_infer.sh --image_dir 图片目录
#   ./run_infer.sh --image_path 单张图片.jpg
#   ./run_infer.sh --image_dir ./test_images --sliding_window --window_width 3072 --window_height 4096 --stride 2000

python infer.py --auto_class_thresholds "$@"
