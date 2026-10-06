# Proyecto Final

## Flujo recomendado en Google Colab

El proyecto está separado en dos notebooks independientes:

1. [`01_single_pixel_camera_colab.ipynb`](01_single_pixel_camera_colab.ipynb) monta Drive, simula la Single Pixel Camera con patrones Hadamard y guarda las vistas reconstruidas junto con las poses COLMAP.
2. [`02_gaussian_splatting_colab.ipynb`](02_gaussian_splatting_colab.ipynb) carga ese resultado desde Drive, prepara las poses, entrena Gaussian Splatting y guarda los renders y métricas.

El primer notebook deja el siguiente contrato de datos:

```text
Mi unidad/inf449-portafolio/proyecto-final/
└── spc_runs/spc_pilot/
    ├── manifest.json
    ├── summary_2d.csv
    ├── original/images/ + sparse/0/
    ├── spc_010/images/ + sparse/0/
    ├── spc_025/images/ + sparse/0/
    └── spc_050/images/ + sparse/0/
```

El Notebook 1 descarga automáticamente `tandt_db.zip` desde Hugging Face en
`/content` usando `wget` y `unzip`; no es necesario subir el dataset a Drive.
Solo se guardan en Drive los resultados persistentes. Para validar el flujo usa
`VIEW_LIMIT = 30`, `IMAGE_SIZE = (64, 64)`, `ITERATIONS = 1500` y una
sola condición (`spc_025`). Después se puede ampliar a todas las vistas y más
iteraciones.

El notebook grande [`notebook.ipynb`](notebook.ipynb) conserva el experimento
completo anterior, incluyendo comparaciones 2D/3D. Los notebooks
[`diagnostico_3dgs.ipynb`](diagnostico_3dgs.ipynb) y
[`gaussian_splatting_colab.ipynb`](gaussian_splatting_colab.ipynb) se mantienen
como material de diagnóstico/referencia.

Uso sugerido:

```text
proyecto-final/
├── informe.pdf
├── presentacion.pdf
└── notebook.ipynb
```

El notebook disponible en [`notebook.ipynb`](notebook.ipynb) está preparado para Google Colab. Monta Drive, descomprime `tandt_db.zip` si es necesario, simula las mediciones de una Single Pixel Camera sobre la escena `truck` para los niveles 5%, 10%, 25% y 50%, genera además un baseline sin SPC y guarda una carpeta compatible con Gaussian Splatting por cada nivel. Las imágenes se reescalan a la resolución original para mantener compatibilidad con las poses COLMAP. También documenta que GPT no participa en la ejecución y muestra la RAM, GPU/VRAM, espacio, Python, PyTorch/CUDA y librerías disponibles en la VM. Las secciones OE4 y OE5 clonan el repositorio oficial de 3DGS, entrenan/renderizan/evalúan las cinco escenas y guardan la comparación 2D vs. 3D en `comparacion_2d_vs_3d.csv`. El notebook además sigue una vista de referencia fija y genera una tira final del caso de estudio desde la imagen original hasta el render 3D.

Si 3DGS no inicia, [`diagnostico_3dgs.ipynb`](diagnostico_3dgs.ipynb) comprueba CUDA, PyTorch, las extensiones compiladas, el formato COLMAP y ejecuta un entrenamiento de solo 10 iteraciones. El entrenamiento principal usa `--resolution 256` en modo piloto y `512` en modo completo para evitar que las imágenes reescaladas a su tamaño original agoten la VRAM.

La preparación 3D alinea también nombres como `01.jpg`, `000001.jpg` o `frame_000001.jpg` mediante su índice numérico final, de modo que las imágenes generadas para la demo mantengan las poses COLMAP correctas.
