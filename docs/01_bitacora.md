# Bitácora

## 2026-11-05 — Exploración inicial

**Hecho**
- Inspección del entorno, datasets, repo oficial, corridas previas y notebooks en `codes/old/`.
  Detalle en `00_plan_y_decisiones.md`.
- Benchmark de la transformada Walsh-Hadamard en GPU con el Python del sistema, porque `.venv` no
  tiene CUDA. Ida y vuelta con error 2.7e-7; coincide con la matriz Sylvester densa (error 1.2e-7).
  - 256×512: 0.9 ms/vista (3 canales)
  - 512×1024: 3.4 ms/vista (3 canales), 0.48 GiB con lote de 16

**Encontrado**
- `.venv` tiene `torch 2.14.1+cpu`, de modo que `torch.cuda.is_available()` da `False`.
- `diff_gaussian_rasterization` y `simple_knn` no están compiladas:
  `ModuleNotFoundError: No module named 'diff_gaussian_rasterization'`.
- `outputs/.../gs_outputs` y `logs` están vacíos: no hay ninguna corrida 3DGS local previa.
- COLMAP no está instalado. El caché de pesos de torch está vacío.

**Pendiente**
- OK del usuario para instalar torch con CUDA, compilar las extensiones y descargar los pesos LPIPS.

## 2026-11-05 — Instalación y compilación (con OK del usuario)

**Hecho**
- `docs/venv_freeze_antes_de_cu128.txt` guarda el `.venv` original y `docs/venv_freeze_despues_de_cu128.txt`
  el estado nuevo.
- `torch 2.14.1+cpu` → `2.11.0+cu128`, `torchvision 0.29.1+cpu` → `0.26.0+cu128`. También `plyfile`, `papermill`.
- `.gitignore`: añadidos `data/`, `outputs/` y `third_party/`.
- Pesos LPIPS: VGG16 (528 MB) y sus capas lineales. **AlexNet no se descargó** porque no hizo falta: se usa la
  implementación LPIPS-VGG del repo, la misma de `metrics.py`.

**Errores y soluciones**

1. Primera compilación de ambas extensiones:
   ```text
   torch/csrc/dynamo/compiled_autograd.h(1143): error C2872: 'std': ambiguous symbol
   error: command '...\nvcc' failed with exit code 2
   ```
   La causa no está en 3DGS. Esa cabecera de torch ya tiene una guarda para Windows
   (`#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))`, línea 1119), pero `CUDAExtension` no define
   `USE_CUDA` en el paso que compila con `cl`.
   **Solución:** `set CL=/DUSE_CUDA` antes de compilar. No hizo falta editar torch ni `third_party/`.
   Tampoco hicieron falta los parches de cabeceras del notebook de Colab (`<cstdint>`, `<float.h>`).
2. `vswhere.exe no se reconoce` al ejecutar `vcvars64.bat` desde `cmd` lanzado por otro proceso: es solo un aviso,
   y `cl` y `nvcc` quedan en el PATH.

**Verificado:** `import diff_gaussian_rasterization, simple_knn._C` funcionan y `distCUDA2` corre en la GPU.

## 2026-11-05 — Runner y notebooks

**Hecho**
- `codes/gs_train_runner.py`, que envuelve `train.training` sin modificar el repo.
- Notebooks 1 y 2 probados de principio a fin con una versión mínima (16 vistas, 300 iteraciones).
- Probada la reanudación: evaluar una corrida ya entrenada, retomar desde un checkpoint y omitir corridas terminadas.

**Errores y soluciones**

1. El parche de `torch.load` del runner se aplica al módulo completo y chocó con `torch.hub`:
   `TypeError: torch.serialization.load() got multiple values for keyword argument 'weights_only'`.
   Solución: usar `weights_only=False` solo como valor por defecto.
2. `TypeError: 'float' object is not callable`: en `report`, el parámetro `elapsed` de `training_report` ocultaba
   mi función del mismo nombre. Se renombró a `clock`.
3. Con `--quiet`, el `safe_state` del repo silencia `stdout`, incluidas las líneas `[ITER n] Evaluating test` que
   alimentan las curvas y mis mensajes de error. El notebook ya no usa `--quiet`, y los errores van a `stderr`.
4. `RuntimeError: view size is not compatible with input tensor's size and stride` en el `psnr` del repo: los tensores
   permutados no son contiguos. Solución: `.contiguous()` en `load_rgb`.
5. `KeyError: '000001'`: `pd.read_csv` convirtió los nombres de vista numéricos en enteros. Afectaría a cualquier
   dataset con nombres así. Solución: `dtype={"view": str}`.
