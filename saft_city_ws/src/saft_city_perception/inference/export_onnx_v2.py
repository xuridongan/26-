#!/usr/bin/env python3
"""ONNX 导出 v2 — 用 Paddle Inference 的 export 接口"""
import os, sys, paddle

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(MODEL_DIR)
sys.path.insert(0, MODEL_DIR)

# 加载模型
from infer import build_infer_model, write_infer_config
from ppdet.core.workspace import create, load_config
from ppdet.utils.checkpoint import load_weight
import yaml

# 构建带 post_process 的完整模型
cfg_path = write_infer_config(0.25, 'model_final.pdparams')
cfg = load_config(str(cfg_path))
model = create(cfg.architecture)
load_weight(model, cfg.weights)
model.eval()

# 用 paddle.jit.to_static 并指定完整的 input_spec
# PP-YOLOE+ 需要 3 个输入: image, im_shape, scale_factor
class Wrapper(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # 只取 bbox 输出
        outs = self.model(x)
        return outs

wrapped = Wrapper(model)

# 构建 dummy 输入
dummy = paddle.randn([1, 3, 640, 640])

# 保存为静态图
paddle.jit.save(wrapped, MODEL_DIR + '/ppdet_simple', input_spec=[dummy])
print("静态图保存成功")

# 转 ONNX
ret = os.system(f"paddle2onnnx --model_dir {MODEL_DIR} --model_filename ppdet_simple.pdmodel --params_filename ppdet_simple.pdiparams --save_file {MODEL_DIR}/model.onnx --opset_version 11 2>&1")
if os.path.exists(MODEL_DIR + '/model.onnx'):
    print(f"ONNX 成功! {os.path.getsize(MODEL_DIR+'/model.onnx')/1024/1024:.1f}MB")
else:
    # paddle2onnx 可能叫 paddle2onnx
    os.system(f"paddle2onnx --model_dir {MODEL_DIR} --model_filename ppdet_simple.pdmodel --params_filename ppdet_simple.pdiparams --save_file {MODEL_DIR}/model.onnx --opset_version 11 2>&1")
    if os.path.exists(MODEL_DIR + '/model.onnx'):
        print(f"ONNX 成功! {os.path.getsize(MODEL_DIR+'/model.onnx')/1024/1024:.1f}MB")
    else:
        print("ONNX 导出失败")
