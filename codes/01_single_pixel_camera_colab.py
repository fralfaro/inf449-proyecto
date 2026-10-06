# %% [markdown]
# # Proyecto final — Notebook 1: Single Pixel Camera
#
# Este notebook está pensado para ejecutarse en Google Colab. Monta Google Drive,
# toma una escena multivista con poses COLMAP, simula mediciones de una Single
# Pixel Camera usando una base de Hadamard y deja las vistas reconstruidas listas
# para el Notebook 2.
#
# La cámara single-pixel se simula sobre imágenes RGB existentes. Por lo tanto,
# el experimento estudia el efecto de la compresión/reconstrucción sobre Gaussian
# Splatting; no sustituye una captura óptica real.

# %% [markdown]
# ## 0. Dependencias
#
# Ejecuta las celdas en orden. La primera corrida puede tardar porque descarga las
# librerías y copia la escena desde Drive a la máquina temporal de Colab.

# %%
!pip -q install numpy scipy pillow scikit-image pandas matplotlib tqdm psutil

# %%
from pathlib import Path
import json
import platform
import shutil
import struct
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.linalg import hadamard
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm

np.random.seed(449)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# %% [markdown]
# ## 1. Google Drive y configuración
#
# El dataset se descarga directamente en `/content`, que es el disco temporal de
# Colab. Los resultados sí se conservan en Drive. El ZIP debe contener la escena
# `truck` con `images/` y `sparse/0/`.

# %%
from google.colab import drive
drive.mount('/content/drive')

DRIVE_PROJECT = Path('/content/drive/MyDrive/inf449-portafolio/proyecto-final')
DATASET_URL = 'https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip'
DATASET_ZIP = Path('/content/tandt_db.zip')
DATASET_DIR = Path('/content/tandt_db_extracted')
SCENE_NAME = 'truck'

RUN_NAME = 'spc_pilot'
RUN_ROOT = DRIVE_PROJECT / 'spc_runs' / RUN_NAME
LOCAL_SCENE = Path('/content') / f'{SCENE_NAME}_source'

# Para la primera ejecución usa 30 vistas y una resolución pequeña. Cuando el
# flujo esté validado, cambia a None y aumenta IMAGE_SIZE si la GPU/CPU lo permite.
VIEW_LIMIT = 30
IMAGE_SIZE = (64, 64)  # debe ser potencia de dos para la base Hadamard
COMPRESSION_RATIOS = [0.10, 0.25, 0.50]
SAVE_RAW_MEASUREMENTS = True

DRIVE_PROJECT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
print('Drive:', DRIVE_PROJECT)
print('Resultados:', RUN_ROOT)

# Descarga local: es bastante más rápido que leer cientos de imágenes desde Drive.
# Si el archivo ya existe en esta sesión, no se vuelve a descargar.
if not DATASET_ZIP.exists():
    print('Descargando dataset desde Hugging Face...')
    !wget -q --show-progress -O "/content/tandt_db.zip" "{DATASET_URL}"
else:
    print('ZIP ya presente:', DATASET_ZIP)

if not DATASET_DIR.exists() or not any(DATASET_DIR.rglob('images.bin')):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print('Descomprimiendo dataset...')
    !unzip -q -o "/content/tandt_db.zip" -d "/content/tandt_db_extracted"

# %% [markdown]
# ## 2. Diagnóstico y localización de la escena

# %%
def gibibytes(value):
    return value / (1024 ** 3)

print('Python:', sys.version.split()[0])
print('Sistema:', platform.platform())
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f'RAM disponible: {gibibytes(memory.available):.2f} GB')
except ImportError:
    pass

def is_scene_folder(path):
    return (path / 'images').is_dir() and (path / 'sparse' / '0').is_dir()

def find_scene(root, scene_name):
    if not root.exists():
        return None

    # Primero buscamos por nombre, tolerando carpetas intermedias como
    # tandt/train/truck o tandt_db/tandt/train/truck.
    for candidate in root.rglob('*'):
        if candidate.is_dir() and candidate.name.casefold() == scene_name.casefold():
            if is_scene_folder(candidate):
                return candidate

    # Fallback: algunos ZIP cambian el nombre de la carpeta, pero conservan la
    # estructura COLMAP. En ese caso usamos la carpeta que contiene images.bin.
    for images_bin in root.rglob('images.bin'):
        candidate = images_bin.parent.parent.parent
        if is_scene_folder(candidate):
            print(f'Usando escena encontrada por estructura COLMAP: {candidate}')
            return candidate
    return None

scene_on_colab = find_scene(DATASET_DIR, SCENE_NAME)

if scene_on_colab is None:
    found = []
    if DATASET_DIR.exists():
        found = [str(path.relative_to(DATASET_DIR))
                 for path in list(DATASET_DIR.rglob('*'))[:40]]
    raise FileNotFoundError(
        f'No encontré {SCENE_NAME!r} dentro de {DATASET_DIR}. '
        f'Contenido inicial: {found}. Revisa la descarga o cambia SCENE_NAME.'
    )

if LOCAL_SCENE.exists():
    shutil.rmtree(LOCAL_SCENE)
shutil.copytree(scene_on_colab, LOCAL_SCENE)
print('Escena descargada en Colab:', scene_on_colab)
print('Copia local de Colab:', LOCAL_SCENE)

# %% [markdown]
# ## 3. Alinear imágenes y poses COLMAP
#
# Gaussian Splatting necesita que cada imagen reconstruida conserve el mismo nombre
# y la misma pose que su registro en `sparse/0/images.bin`.

# %%
def read_colmap_image_names(images_bin):
    names = []
    with open(images_bin, 'rb') as file:
        count = struct.unpack('<Q', file.read(8))[0]
        for _ in range(count):
            header = file.read(68)
            if len(header) != 68:
                raise RuntimeError('images.bin está truncado.')
            name_bytes = bytearray()
            while True:
                byte = file.read(1)
                if byte == b'\x00':
                    break
                if not byte:
                    raise RuntimeError('No terminó el nombre de una imagen COLMAP.')
                name_bytes.extend(byte)
            names.append(name_bytes.decode('utf-8'))
            n_points = struct.unpack('<Q', file.read(8))[0]
            file.seek(24 * n_points, 1)
    return names

images_dir = LOCAL_SCENE / 'images'
images_bin = LOCAL_SCENE / 'sparse' / '0' / 'images.bin'
pose_names = read_colmap_image_names(images_bin)
available = {path.name.casefold(): path for path in images_dir.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}

aligned = []
missing = []
for pose_name in pose_names:
    path = available.get(Path(pose_name).name.casefold())
    if path is None:
        missing.append(pose_name)
    else:
        aligned.append((path, Path(pose_name).name))

if not aligned:
    raise RuntimeError('Ninguna imagen de images/ coincide con images.bin.')
if missing:
    print(f'Advertencia: {len(missing)} poses no tienen imagen y se ignorarán.')

if VIEW_LIMIT is None or VIEW_LIMIT >= len(aligned):
    selected = aligned
else:
    indices = np.unique(np.linspace(0, len(aligned) - 1, VIEW_LIMIT, dtype=int))
    selected = [aligned[index] for index in indices]

print(f'Vistas disponibles: {len(aligned)} | seleccionadas: {len(selected)}')
print('Ejemplo:', selected[:3])

def load_rgb(path, size=None):
    with Image.open(path) as image:
        image = image.convert('RGB')
        if size is not None:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0

def save_rgb(array, path, size=None):
    array = np.clip(array, 0, 1)
    image = Image.fromarray(np.round(array * 255).astype(np.uint8), mode='RGB')
    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(path)

def metrics(reference, reconstruction):
    return {
        'psnr_db': float(peak_signal_noise_ratio(reference, reconstruction, data_range=1.0)),
        'ssim': float(structural_similarity(reference, reconstruction, channel_axis=-1, data_range=1.0)),
    }

# %% [markdown]
# ## 4. Medición y reconstrucción Single Pixel
#
# Usamos una transformación Hadamard separable. Cada coeficiente equivale a una
# medición con una máscara `+1/-1`; conservar solo los coeficientes de baja
# frecuencia permite controlar el ratio de mediciones.

# %%
def make_hadamard_order(size):
    if size & (size - 1):
        raise ValueError('IMAGE_SIZE debe ser potencia de dos.')
    basis = hadamard(size).astype(np.float32) / np.sqrt(size)
    changes = np.sum(basis[:, 1:] != basis[:, :-1], axis=1)
    order_1d = np.argsort(changes, kind='stable')
    pairs = [(i, j) for i in order_1d for j in order_1d]
    pairs.sort(key=lambda pair: (changes[pair[0]] + changes[pair[1]],
                                 max(changes[pair[0]], changes[pair[1]])))
    return basis, pairs

H, HADAMARD_ORDER = make_hadamard_order(IMAGE_SIZE[0])

def spc_reconstruct(image, ratio):
    coefficients = np.stack([H @ image[..., channel] @ H.T
                              for channel in range(image.shape[-1])], axis=-1)
    count = max(1, int(round(float(ratio) * len(HADAMARD_ORDER))))
    selected = HADAMARD_ORDER[:count]
    measurements = np.asarray([coefficients[i, j] for i, j in selected], dtype=np.float32)
    sparse_coefficients = np.zeros_like(coefficients)
    for value, (i, j) in zip(measurements, selected):
        sparse_coefficients[i, j] = value
    reconstruction = np.stack([H.T @ sparse_coefficients[..., channel] @ H
                               for channel in range(image.shape[-1])], axis=-1)
    return np.clip(reconstruction, 0, 1), measurements

reference_path, reference_pose_name = selected[len(selected) // 2]
reference_small = load_rgb(reference_path, IMAGE_SIZE)
fig, axes = plt.subplots(1, len(COMPRESSION_RATIOS) + 1, figsize=(4 * (len(COMPRESSION_RATIOS) + 1), 4))
axes[0].imshow(reference_small)
axes[0].set_title('Vista reducida')
axes[0].axis('off')
for axis, ratio in zip(axes[1:], COMPRESSION_RATIOS):
    reconstruction, measurements = spc_reconstruct(reference_small, ratio)
    axis.imshow(reconstruction)
    axis.set_title(f'SPC {ratio:.0%}\n{len(measurements)} mediciones')
    axis.axis('off')
plt.tight_layout()

# %% [markdown]
# ## 5. Generar el dataset de salida en Drive
#
# Cada condición contiene `images/` y una copia de `sparse/`, de modo que el
# Notebook 2 pueda ejecutarse de forma independiente. Las imágenes SPC se
# reconstruyen a 64×64 y luego se escalan a la resolución original de cada vista;
# así no se invalidan las intrínsecas calibradas por COLMAP.

# %%
conditions = ['original'] + [f'spc_{int(round(ratio * 100)):03d}' for ratio in COMPRESSION_RATIOS]
condition_dirs = {}
for condition in conditions:
    condition_dir = RUN_ROOT / condition
    if condition_dir.exists():
        shutil.rmtree(condition_dir)
    (condition_dir / 'images').mkdir(parents=True, exist_ok=True)
    shutil.copytree(LOCAL_SCENE / 'sparse', condition_dir / 'sparse')
    condition_dirs[condition] = condition_dir

rows = []
measurement_root = RUN_ROOT / 'measurements'
if SAVE_RAW_MEASUREMENTS:
    measurement_root.mkdir(parents=True, exist_ok=True)

started = time.perf_counter()
for source_path, pose_name in tqdm(selected, desc='Generando vistas'):
    original = load_rgb(source_path)
    working = load_rgb(source_path, IMAGE_SIZE)
    original_size = (original.shape[1], original.shape[0])

    original_output = condition_dirs['original'] / 'images' / pose_name
    shutil.copy2(source_path, original_output)
    rows.append({'condition': 'original', 'image_name': pose_name,
                 'ratio': 1.0, **metrics(original, original)})

    for ratio in COMPRESSION_RATIOS:
        condition = f'spc_{int(round(ratio * 100)):03d}'
        reconstruction, measurements = spc_reconstruct(working, ratio)
        output = condition_dirs[condition] / 'images' / pose_name
        save_rgb(reconstruction, output, size=original_size)
        reconstructed_full = load_rgb(output)
        row = {'condition': condition, 'image_name': pose_name,
               'ratio': ratio, 'num_measurements': len(measurements),
               **metrics(original, reconstructed_full)}
        rows.append(row)
        if SAVE_RAW_MEASUREMENTS:
            target = measurement_root / condition
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / f'{Path(pose_name).stem}.npz',
                                measurements=measurements, ratio=ratio,
                                image_name=pose_name)

summary = pd.DataFrame(rows)
summary.to_csv(RUN_ROOT / 'metrics_2d_per_view.csv', index=False)
summary_mean = (summary.groupby(['condition', 'ratio'], as_index=False)
                .agg(num_views=('image_name', 'nunique'),
                     psnr_db_mean=('psnr_db', 'mean'),
                     psnr_db_std=('psnr_db', 'std'),
                     ssim_mean=('ssim', 'mean'),
                     ssim_std=('ssim', 'std')))
summary_mean.to_csv(RUN_ROOT / 'summary_2d.csv', index=False)

manifest = {
    'scene_name': SCENE_NAME,
    'run_name': RUN_NAME,
    'image_size_for_spc': list(IMAGE_SIZE),
    'compression_ratios': COMPRESSION_RATIOS,
    'view_limit': VIEW_LIMIT,
    'num_views': len(selected),
    'conditions': conditions,
    'reference_view': reference_pose_name,
    'source_scene': str(scene_on_colab),
    'note': 'SPC simulada con patrones Hadamard sobre vistas RGB multivista; poses COLMAP conservadas.',
}
(RUN_ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f'Tiempo de generación: {time.perf_counter() - started:.2f} s')
display(summary_mean)

# %% [markdown]
# ## 6. Verificación del contrato para Gaussian Splatting

# %%
for condition, condition_dir in condition_dirs.items():
    images = list((condition_dir / 'images').glob('*'))
    if len(images) != len(selected):
        raise RuntimeError(f'{condition}: se esperaban {len(selected)} imágenes y hay {len(images)}.')
    if not (condition_dir / 'sparse' / '0' / 'images.bin').exists():
        raise RuntimeError(f'{condition}: falta sparse/0/images.bin.')
    print(f'{condition}: {len(images)} imágenes + poses COLMAP OK')

print('\nListo. El Notebook 2 debe usar:')
print(RUN_ROOT)
print('Condiciones:', conditions)
