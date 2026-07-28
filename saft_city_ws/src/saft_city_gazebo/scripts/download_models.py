#!/usr/bin/env python3
"""下载Gazebo官方模型到本地"""
import os, urllib.request, re

MODELS_DIR = '/home/xuan/saft_city_ws/src/saft_city_gazebo/models/osrf_models'

NEEDED = [
    'person_standing', 'person_walking',
    'house_1', 'house_2', 'house_3', 'office_building',
    'dumpster', 'first_2015_trash_can',
    'oak_tree', 'pine_tree', 'bush',
    'construction_barrel', 'construction_cone',
    'jersey_barrier', 'lamp_post', 'mailbox', 'fire_hydrant',
    'cafe_table', 'bench',
]

BASE_URL = 'https://raw.githubusercontent.com/osrf/gazebo_models/master'
TIMEOUT = 15

def dl(url, path):
    """下载文件，自动创建目录"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        return False

def get_file(url, path):
    """下载一个文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as f:
            data = f.read()
        with open(path, 'wb') as out:
            out.write(data)
        return True, data
    except Exception as e:
        return False, str(e)

for idx, model_name in enumerate(NEEDED):
    base_path = os.path.join(MODELS_DIR, model_name)
    config_path = os.path.join(base_path, 'model.config')
    sdf_path = os.path.join(base_path, 'model.sdf')

    if os.path.exists(config_path) and os.path.exists(sdf_path):
        print(f"[{idx+1}/{len(NEEDED)}] ✓ {model_name}")
        continue

    print(f"[{idx+1}/{len(NEEDED)}] {model_name}...")

    # 下载 model.config
    ok, _ = get_file(f'{BASE_URL}/{model_name}/model.config', config_path)
    if not ok:
        print(f"  跳过")
        continue
    print(f"  model.config ✓")

    # 下载 model.sdf
    ok, sdf_data = get_file(f'{BASE_URL}/{model_name}/model.sdf', sdf_path)
    if not ok:
        print(f"  model.sdf ✗")
        continue
    print(f"  model.sdf ✓")

    # 解析 SDF 找 mesh 和材质依赖
    if sdf_data:
        sdf_text = sdf_data.decode('utf-8')

        # 网格文件
        for uri in re.findall(r'<uri>(.*?)</uri>', sdf_text):
            if uri.startswith('model://'): continue
            mesh_url = f'{BASE_URL}/{model_name}/{uri}'
            mesh_path = os.path.join(base_path, uri)
            if dl(mesh_url, mesh_path):
                print(f"  mesh: {os.path.basename(uri)} ✓")

        # 材质文件
        for m in re.finditer(r'<uri>(.*?)</uri>.*?<name>(.*?)</name>', sdf_text, re.DOTALL):
            mat_uri = m.group(1).strip()
            if mat_uri.startswith('model://'): continue
            # 下载材质脚本
            mat_path = os.path.join(base_path, mat_uri)
            ok, mat_data = get_file(f'{BASE_URL}/{model_name}/{mat_uri}', mat_path)
            if ok and mat_data:
                print(f"  material ✓")
                # 找纹理
                for tex in re.findall(r'texture\s+(\S+)', mat_data.decode('utf-8')):
                    tex_url = f'{BASE_URL}/{model_name}/{tex}'
                    tex_path = os.path.join(base_path, tex)
                    if dl(tex_url, tex_path):
                        print(f"    texture: {os.path.basename(tex)} ✓")

    print(f"  完成")

# 创建 osrf 模型索引文件，让 Gazebo 能找到它们
with open(os.path.join(MODELS_DIR, 'osrf_models.marker'), 'w') as f:
    f.write('OSRF models downloaded')

print(f"\n✅ 全部完成！模型已下载到: {MODELS_DIR}")
print(f"将以下路径加入 GAZEBO_MODEL_PATH:")
print(f"  {MODELS_DIR}")
