# %% [markdown]
# # Proyecto final — Notebook 2: Gaussian Splatting
#
# Este notebook se ejecuta en Google Colab, monta Drive y consume el resultado
# generado por `01_single_pixel_camera_colab.ipynb`. Entrena Gaussian Splatting
# con las vistas originales y/o reconstruidas por la Single Pixel Camera.
#
# `/content` es temporal. Los modelos, logs y métricas se copian a Drive al final
# de cada condición.

# %% [markdown]
# ## 0. Configuración de Drive

# %%
!pip -q install plyfile pandas matplotlib pillow

# %%
from pathlib import Path
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

DRIVE_PROJECT = Path('/content/drive/MyDrive/inf449-portafolio/proyecto-final')
RUN_ROOT = DRIVE_PROJECT / 'spc_runs' / 'spc_pilot'

# Empieza con una sola condición. Después puedes agregar 'original' y 'spc_010'.
TRAIN_CONDITIONS = ['spc_025']
ITERATIONS = 1500
GS_RESOLUTION = 256
DATA_DEVICE = 'cpu'
RETRAIN = True

LOCAL_INPUT_ROOT = Path('/content/gs_inputs')
LOCAL_OUTPUT_ROOT = Path('/content/gs_outputs')
DRIVE_OUTPUT_ROOT = RUN_ROOT / 'gaussian_splatting_outputs'
GS_REPO = Path('/content/gaussian-splatting')
GS_REPO_URL = 'https://github.com/camenduru/gaussian-splatting'

manifest_path = RUN_ROOT / 'manifest.json'
if not manifest_path.exists():
    raise FileNotFoundError(
        f'No existe {manifest_path}. Ejecuta primero el Notebook 1 o corrige RUN_ROOT.'
    )
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
print('Escena:', manifest['scene_name'])
print('Condiciones disponibles:', manifest['conditions'])
print('Condiciones seleccionadas:', TRAIN_CONDITIONS)

for condition in TRAIN_CONDITIONS:
    if condition not in manifest['conditions']:
        raise ValueError(f'{condition} no existe en el resultado del Notebook 1.')

# %% [markdown]
# ## 1. Diagnóstico de la GPU

# %%
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA disponible:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        capability = torch.cuda.get_device_capability(0)
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{capability[0]}.{capability[1]}'
        print('TORCH_CUDA_ARCH_LIST:', os.environ['TORCH_CUDA_ARCH_LIST'])
    else:
        raise RuntimeError('Gaussian Splatting requiere una sesión Colab con GPU CUDA.')
except ImportError as error:
    raise RuntimeError('PyTorch no está disponible en este runtime de Colab.') from error

# %% [markdown]
# ## 2. Copiar las entradas desde Drive y comprobar imágenes/poses

# %%
if LOCAL_INPUT_ROOT.exists():
    shutil.rmtree(LOCAL_INPUT_ROOT)
LOCAL_INPUT_ROOT.mkdir(parents=True, exist_ok=True)

local_conditions = {}
for condition in TRAIN_CONDITIONS:
    source = RUN_ROOT / condition
    destination = LOCAL_INPUT_ROOT / condition
    shutil.copytree(source, destination)
    images = list((destination / 'images').glob('*'))
    sparse = destination / 'sparse' / '0'
    if not images or not (sparse / 'images.bin').exists():
        raise RuntimeError(f'{condition}: faltan imágenes o sparse/0/images.bin.')
    local_conditions[condition] = destination
    print(f'{condition}: {len(images)} imágenes listas en {destination}')

# %% [markdown]
# ## 3. Descargar y compilar Gaussian Splatting
#
# El repositorio usa extensiones CUDA (`diff-gaussian-rasterization` y
# `simple-knn`). La compilación puede tardar varios minutos la primera vez.

# %%
def run_command(command, cwd=None, label=None, log_path=None, check=True):
    command = [str(part) for part in command]
    print('\n$', ' '.join(command))
    completed = subprocess.run(command, cwd=cwd, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = completed.stdout or ''
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(output, encoding='utf-8')
    print(output[-12000:])
    if check and completed.returncode != 0:
        raise RuntimeError(f'Falló {label or command[0]} con código {completed.returncode}.')
    return completed

if not GS_REPO.exists():
    run_command(['git', 'clone', '--recursive', GS_REPO_URL, GS_REPO], label='clonar 3DGS')
else:
    run_command(['git', 'submodule', 'update', '--init', '--recursive'], cwd=GS_REPO,
                label='inicializar submódulos')

# Parches idempotentes que evitan errores de cabeceras en algunos runtimes Colab.
rasterizer_header = GS_REPO / 'submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h'
if rasterizer_header.exists():
    text = rasterizer_header.read_text(encoding='utf-8')
    if '#include <cstdint>' not in text:
        rasterizer_header.write_text(text.replace('#pragma once', '#pragma once\n#include <cstdint>', 1), encoding='utf-8')

knn_source = GS_REPO / 'submodules/simple-knn/simple_knn.cu'
if knn_source.exists():
    text = knn_source.read_text(encoding='utf-8')
    if '#include <float.h>' not in text:
        knn_source.write_text('#include <float.h>\n' + text, encoding='utf-8')

run_command([sys.executable, '-m', 'pip', 'install', '--no-build-isolation', 'plyfile'], label='instalar plyfile')
run_command([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
             GS_REPO / 'submodules/diff-gaussian-rasterization'], label='compilar rasterizer')
run_command([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
             GS_REPO / 'submodules/simple-knn'], label='compilar simple-knn')

from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2
print('Extensiones CUDA importadas correctamente.')

# %% [markdown]
# ## 4. Filtrar `images.bin` para las vistas realmente generadas
#
# El dataset original puede tener más vistas que el piloto. Esta celda conserva
# solo las poses correspondientes a las imágenes presentes en cada condición.

# %%
def read_colmap_records(path):
    records = []
    with open(path, 'rb') as file:
        count = struct.unpack('<Q', file.read(8))[0]
        for _ in range(count):
            header = file.read(68)
            if len(header) != 68:
                raise RuntimeError(f'images.bin truncado: {path}')
            name_bytes = bytearray()
            while True:
                byte = file.read(1)
                if byte == b'\x00':
                    break
                name_bytes.extend(byte)
            n_points = struct.unpack('<Q', file.read(8))[0]
            points = file.read(24 * n_points)
            records.append((header, name_bytes.decode('utf-8'), n_points, points))
    return records

def write_colmap_records(path, records):
    with open(path, 'wb') as file:
        file.write(struct.pack('<Q', len(records)))
        for header, name, n_points, points in records:
            file.write(header)
            file.write(name.encode('utf-8') + b'\x00')
            file.write(struct.pack('<Q', n_points))
            file.write(points)

def filter_poses(scene_dir):
    images_dir = scene_dir / 'images'
    images_bin = scene_dir / 'sparse' / '0' / 'images.bin'
    available = {path.name.casefold() for path in images_dir.iterdir() if path.is_file()}
    records = read_colmap_records(images_bin)
    selected = [record for record in records if Path(record[1]).name.casefold() in available]
    if not selected:
        raise RuntimeError(f'{scene_dir}: ninguna imagen coincide con las poses COLMAP.')
    write_colmap_records(images_bin, selected)
    return len(records), len(selected)

for condition, scene_dir in local_conditions.items():
    before, after = filter_poses(scene_dir)
    print(f'{condition}: poses COLMAP {before} -> {after}')

# %% [markdown]
# ## 5. Entrenar, renderizar y evaluar

# %%
LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
records = []

for condition, scene_dir in local_conditions.items():
    local_model = LOCAL_OUTPUT_ROOT / f'gs_{condition}'
    drive_model = DRIVE_OUTPUT_ROOT / f'gs_{condition}'
    train_log = DRIVE_OUTPUT_ROOT / f'{condition}_train.log'
    render_log = DRIVE_OUTPUT_ROOT / f'{condition}_render.log'
    metrics_log = DRIVE_OUTPUT_ROOT / f'{condition}_metrics.log'
    result_json = local_model / 'results.json'

    if RETRAIN or not result_json.exists():
        started = time.perf_counter()
        run_command([
            sys.executable, 'train.py', '-s', scene_dir, '-m', local_model,
            '--eval', '--iterations', ITERATIONS,
            '--test_iterations', ITERATIONS, '--save_iterations', ITERATIONS,
            '--resolution', GS_RESOLUTION, '--data_device', DATA_DEVICE,
        ], cwd=GS_REPO, label=f'entrenar {condition}', log_path=train_log)
        train_seconds = time.perf_counter() - started
    else:
        train_seconds = 0.0
        print(f'{condition}: se reutiliza {local_model}')

    run_command([sys.executable, 'render.py', '-m', local_model,
                 '--iteration', ITERATIONS], cwd=GS_REPO,
                label=f'renderizar {condition}', log_path=render_log)
    run_command([sys.executable, 'metrics.py', '-m', local_model], cwd=GS_REPO,
                label=f'evaluar {condition}', log_path=metrics_log)

    if drive_model.exists():
        shutil.rmtree(drive_model)
    shutil.copytree(local_model, drive_model)
    records.append({'condition': condition, 'iterations': ITERATIONS,
                    'train_seconds': train_seconds,
                    'model_on_drive': str(drive_model),
                    'status': 'ok'})
    print(f'{condition}: modelo copiado a {drive_model}')

status = pd.DataFrame(records)
status.to_csv(DRIVE_OUTPUT_ROOT / 'training_status.csv', index=False)
display(status)

# %% [markdown]
# ## 6. Inspeccionar renders y consolidar métricas

# %%
def metric_blocks(node):
    found = []
    if isinstance(node, dict):
        upper = {str(key).upper(): value for key, value in node.items()}
        if 'PSNR' in upper and 'SSIM' in upper:
            found.append({
                'psnr_3d_db': upper.get('PSNR'),
                'ssim_3d': upper.get('SSIM'),
                'lpips_3d': upper.get('LPIPS'),
            })
        for value in node.values():
            found.extend(metric_blocks(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(metric_blocks(value))
    return found

metric_rows = []
for condition in TRAIN_CONDITIONS:
    model = DRIVE_OUTPUT_ROOT / f'gs_{condition}'
    result_path = model / 'results.json'
    if result_path.exists():
        blocks = metric_blocks(json.loads(result_path.read_text(encoding='utf-8')))
        if blocks:
            metric_rows.append({'condition': condition, **blocks[-1]})
    render_dir = model / 'test' / f'ours_{ITERATIONS}' / 'renders'
    render_files = sorted(render_dir.glob('*'))
    if render_files:
        fig, axes = plt.subplots(1, min(3, len(render_files)), figsize=(15, 4))
        axes = np.atleast_1d(axes)
        for axis, render_file in zip(axes, render_files[:3]):
            axis.imshow(Image.open(render_file).convert('RGB'))
            axis.set_title(render_file.name)
            axis.axis('off')
        fig.suptitle(f'Renders Gaussian Splatting — {condition}')
        plt.tight_layout()

metrics_3d = pd.DataFrame(metric_rows)
metrics_3d.to_csv(DRIVE_OUTPUT_ROOT / 'metrics_3d.csv', index=False)
display(metrics_3d)

summary_2d_path = RUN_ROOT / 'summary_2d.csv'
if summary_2d_path.exists():
    summary_2d = pd.read_csv(summary_2d_path)
    display(summary_2d)
    print('Comparación: summary_2d.csv contiene calidad de entrada; metrics_3d.csv contiene calidad del render.')

print('\nProceso terminado.')
print('Resultados persistentes en:', DRIVE_OUTPUT_ROOT)
