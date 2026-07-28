#!/usr/bin/env python3
"""导出 PP-YOLOE+ 模型为 ONNX（去掉 NMS，后处理在 Python 中做）"""
import os, sys, paddle

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(MODEL_DIR)
sys.path.insert(0, MODEL_DIR)

from ppdet.core.workspace import create, load_config
from ppdet.utils.checkpoint import load_weight
from infer import write_infer_config, CLASS_NAMES

print("加载模型...")
cfg_path = write_infer_config(0.01, 'model_final.pdparams')
cfg = load_config(str(cfg_path))
model = create(cfg.architecture)
model.post_process = None
load_weight(model, cfg.weights)
model.eval()

# 去掉 NMS 头，只输出原始预测
model.yolo_head.nms = None

# 转为静态图
dummy = paddle.randn([1, 3, 640, 640])
im_shape = paddle.to_tensor([[480, 640]], dtype='float32')
scale = paddle.to_tensor([[1.0, 1.0]], dtype='float32')

model = paddle.jit.to_static(model, input_spec=[dummy, im_shape, scale])
paddle.jit.save(model, MODEL_DIR + '/ppdet_model_nms')
print("静态图保存成功")

os.system(f"paddle2onnx --model_dir {MODEL_DIR} --model_filename ppdet_model_nms.pdmodel --params_filename ppdet_model_nms.pdiparams --save_file {MODEL_DIR}/model.onnx --opset_version 11 2>&1")

if os.path.exists(MODEL_DIR + '/model.onnx'):
    print(f"ONNX 模型已生成: {MODEL_DIR}/model.onnx ({os.path.getsize(MODEL_DIR+'/model.onnx')/1024/1024:.1f}MB)")
else:
    print("ONNX 转换失败")
