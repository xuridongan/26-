@echo off
REM 推理脚本 - 安全城市场景检测
REM 用法:
REM   run_infer.bat --image_dir 图片目录
REM   run_infer.bat --image_path 单张图片.jpg

python infer.py --auto_class_thresholds %*
