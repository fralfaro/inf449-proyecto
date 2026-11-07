# Códigos del proyecto

| Archivo | Contenido |
|---|---|
| `01_spc_simulacion_reconstruccion.ipynb` | Pasos 1 y 2: mediciones SPC, reconstrucción y métricas 2D |
| `02_gaussian_splatting.ipynb` | Pasos 3 y 4: escenas COLMAP, entrenamiento 3DGS y comparación |
| `gs_train_runner.py` | Entrena y evalúa una escena en un proceso propio, con reanudación y registro de VRAM y temperatura |
| `gaussian_splatting_colab.ipynb` | Referencia original del flujo de compilación de 3DGS en Colab |
| `3d_gaussian_splatting_low.pdf` | Paper de 3DGS (Kerbl et al., 2023) |
| `old/` | Intentos previos, solo como referencia. Las lecciones están en `docs/00_plan_y_decisiones.md` |

## Requisitos

Probado en Windows 11 con RTX 4060 Laptop (8 GB), Python 3.11, Visual Studio 2022 (MSVC 14.42) y CUDA Toolkit 12.8.

Los datasets van en `data/tandt_db/` y el repo oficial, con sus submódulos, en `third_party/gaussian-splatting/`.

## Instalación

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Para compilar las extensiones CUDA de 3DGS, ejecuta esto en `cmd`, no en PowerShell, desde la raíz del repo:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set DISTUTILS_USE_SDK=1
set CL=/DUSE_CUDA
uv pip install --python .venv\Scripts\python.exe --no-build-isolation third_party\gaussian-splatting\submodules\simple-knn
uv pip install --python .venv\Scripts\python.exe --no-build-isolation third_party\gaussian-splatting\submodules\diff-gaussian-rasterization
```

`CL=/DUSE_CUDA` es necesario: sin esa definición, el build falla en una cabecera de torch con
`compiled_autograd.h(1143): error C2872: 'std': ambiguous symbol`. Ver `docs/01_bitacora.md`.

Los pesos de LPIPS-VGG (unos 530 MB) se descargan solos la primera vez, al caché de torch.

## Ejecución

Abre los notebooks con el kernel de `.venv` y ejecútalos en orden. Los parámetros están en la primera celda
de código de cada uno. Las rutas son relativas a la raíz del repo.

Sin abrir Jupyter, desde la raíz del repo:

```powershell
.venv\Scripts\python.exe -m papermill --cwd codes codes\01_spc_simulacion_reconstruccion.ipynb salida_01.ipynb -p NUM_VIEWS 32 -p RUN_NAME prueba
.venv\Scripts\python.exe -m papermill --cwd codes codes\02_gaussian_splatting.ipynb salida_02.ipynb -p SPC_RUN prueba -p GS_RUN prueba
```

**Reanudación.** Volver a ejecutar un notebook con el mismo nombre de corrida omite lo terminado. Las
corridas de 3DGS interrumpidas siguen desde su último checkpoint.

## Salidas

```text
outputs/
├── spc_runs/<escena>/<corrida>/
│   ├── manifest.json               # contrato de datos para el Notebook 2
│   ├── summary_2d.csv              # métricas 2D por condición
│   ├── lambda_search.csv           # búsqueda de lambda para TV
│   ├── estudio_*.csv               # estudios complementarios
│   ├── control_resample/           # solo remuestreo, sin SPC
│   └── <patrón>_r<ratio>/
│       ├── config.json
│       ├── measurements/*.npz      # códigos del sensor; las máscaras se regeneran desde la semilla
│       └── <método>/
│           ├── images/*.jpg        # mismos nombres que las poses COLMAP
│           └── metrics_per_view.csv
└── gs_runs/<escena>/<corrida>/
    ├── run_config.json             # parámetros y fecha
    ├── sanity_checks.csv
    ├── resultados_3dgs.csv, .json
    ├── baseline_vs_paper.csv
    ├── propagacion_2d_3d.csv
    ├── figures/
    ├── logs/<condición>.log
    ├── scenes/<condición>/         # images/ + sparse/0/
    └── models/<condición>/
        ├── metrics.json, run_status.json
        ├── train_log.csv           # pérdida, gaussianas y VRAM cada 100 iteraciones
        ├── gpu_log_*.csv           # temperatura, potencia y memoria
        ├── per_view_<iter>.csv
        ├── test_renders/iteration_<iter>/*.png
        ├── point_cloud/iteration_<iter>/point_cloud.ply
        └── chkpnt<iter>.pth        # checkpoint final
```

Las corridas previas en `outputs/spc_runs/spc_local/` y `outputs/proyecto_final_resultados/` se conservan sin cambios.
