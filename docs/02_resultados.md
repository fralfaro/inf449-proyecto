# Resultados, estado y pendientes

Este archivo reúne lo hecho, los resultados y lo que falta. Está pensado para pasarlo después al `README.md`.
Convención: toda cifra proviene de un CSV en `outputs/`; la fuente va indicada.

## 1. Qué se hizo

| Área | Estado | Dónde |
|---|---|---|
| Entorno local (torch 2.11+cu128, extensiones CUDA de 3DGS compiladas) | Hecho | `docs/01_bitacora.md`, `codes/README.md` |
| Simulación SPC (Hadamard ordenado, binario aleatorio, ruido tipo DMD) | Hecho | `codes/01_spc_simulacion_reconstruccion.ipynb` |
| Reconstrucción directa y TV-FISTA, métricas PSNR/SSIM/LPIPS | Hecho | mismo notebook |
| Estudios complementarios (patrones, ruido, color, Bernoulli) | Hecho en pocas vistas | `estudio_*.csv` |
| Entrenamiento 3DGS (runner con reanudación, notebook 02) | **Implementado, probado en versión mínima; sin resultados** | `codes/02_gaussian_splatting.ipynb`, `codes/gs_train_runner.py` |
| Paper IEEE (esqueleto en español, 17 referencias por verificar) | Esqueleto; faltan resultados | `latex/informe/conference_101719.tex` |
| Alcance y reglas del proyecto | Hecho | `CLAUDE.md` |

Alcance acordado: **el paper cubre solo 2D**. El 3D queda implementado y documentado para continuarlo después.

## 2. Resultados 2D

### 2.1 Corrida definitiva (`paper24`)

4 escenas (Truck, Train, DrJohnson, Playroom) × 24 vistas equiespaciadas, rejilla 512×1024, razones 5/10/25/50 %,
patrones `hadamard_cc` y `random`, métodos directo y TV, semilla 449.

Fuente: `outputs/spc_runs/<escena>/paper24/summary_2d.csv`.

Corrida terminada el 2026-11-07: Truck 12.5 min, Train 12.2, DrJohnson 15.1, Playroom 13.1 (53 min en total, RTX 4060 Laptop).
Datos agregados: `outputs/spc_runs/resumen_paper24_todas_escenas.csv` y `resumen_paper24_media_escenas.csv`
(se regeneran con `outputs/notebooks_ejecutados/figuras_paper.py`).

**Media sobre las 4 escenas** (96 vistas; PSNR en dB):

| Patrón · método | 5 % | 10 % | 25 % | 50 % |
|---|---:|---:|---:|---:|
| Hadamard · directa | 26.25 | 28.26 | 31.07 | 32.58 |
| **Hadamard · TV** | **28.05** | **30.11** | **32.95** | **34.63** |
| Aleatorio · directa | 13.62 | 13.93 | 14.94 | 17.10 |
| Aleatorio · TV | 26.52 | 28.56 | 31.37 | 33.44 |
| Control (solo remuestreo) | 41.04 | | | |

SSIM y LPIPS de Hadamard + TV: 0.798 / 0.358 (5 %), 0.846 / 0.286 (10 %), 0.902 / 0.193 (25 %), 0.914 / 0.152 (50 %).

**Hadamard + TV por escena** (PSNR, dB):

| Escena | 5 % | 10 % | 25 % | 50 % | Control |
|---|---:|---:|---:|---:|---:|
| Truck | 24.11 | 26.13 | 29.55 | 32.10 | 38.07 |
| Train | 23.92 | 25.81 | 29.08 | 31.81 | 37.70 |
| DrJohnson | 32.64 | 34.81 | 36.96 | 37.18 | 44.88 |
| Playroom | 31.54 | 33.70 | 36.20 | 37.44 | 43.52 |

Lectura:
- La TV mejora a la directa con Hadamard en 1.80, 1.86, 1.88 y 2.05 dB (5, 10, 25, 50 %).
- Con aleatorio, la directa no sirve (13.6–17.1 dB); con TV llega a 26.5–33.4 dB y se acerca a Hadamard + TV a razones altas.
- La escena influye más que el método (≈ 8 dB entre Train y DrJohnson al 5 %); el control muestra la misma separación.
  Hipótesis sin verificar: Deep Blending tiene mayor resolución nativa que la rejilla 512×1024 y se suaviza al reducirla.
- Tiempos: directa 0.02–0.04 s por vista; TV 0.15–1.3 s por vista.

**Estudios complementarios** (Truck, 3 vistas, 25 %; solo indicativos), de `estudio_*.csv`:
- Ruido (Hadamard + TV): ideal 29.09 dB, 1e9 fotones 28.55, 1e8 26.98, 1e7 19.85; luz lineal 27.89.
- Orden (directa, 25 %): cake-cutting 27.27, Russian dolls 26.83, Fourier 26.47.
- Color (presupuesto 0.75N, directa): por canal 27.27 frente a Bayer 25.57 con Hadamard; con aleatorio 13.62 frente a 19.49.
- Matriz aleatoria estructurada frente a Bernoulli denso (baja resolución): PSNR casi idéntico (24.39 y 24.31 al 25 %).

### 2.2 Corrida de prueba (`prueba_v32`, Truck, 32 vistas) — provisional

Solo para orientar; no va al paper. Fuente: `outputs/spc_runs/truck/prueba_v32/summary_2d.csv`, vistas de test (n = 4).

| Condición | Razón | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Control (solo remuestreo) | 100 % | 38.15 | 0.985 | 0.0027 |
| Hadamard + directa | 5 % | 23.04 | 0.633 | 0.506 |
| Hadamard + TV | 5 % | 24.14 | 0.709 | 0.442 |
| Hadamard + directa | 25 % | 28.07 | 0.823 | 0.249 |
| Hadamard + TV | 25 % | 29.41 | 0.874 | 0.182 |
| Hadamard + directa | 50 % | 30.80 | 0.850 | 0.164 |
| Hadamard + TV | 50 % | 31.99 | 0.898 | 0.123 |
| Aleatorio + directa | 25 % | 14.43 | 0.216 | 0.670 |
| Aleatorio + TV | 25 % | 28.30 | 0.812 | 0.291 |

Lectura provisional:
- El control a 38 dB indica que el cuello de botella a razones altas es el remuestreo a la rejilla 512×1024 y el JPEG.
- Con Hadamard ordenado, TV mejora a la directa en cerca de 1–1.5 dB en todos los ratios.
- Con patrones aleatorios, la directa es inservible (13–16 dB) y TV recupera 23–31 dB, así que la regularización es imprescindible.
- A igual razón, Hadamard + TV supera al aleatorio + TV por poco en 25 %, y por más en 5 %.

### 2.3 Del piloto 3D incompleto

`outputs/gs_runs/truck/prueba_v251` llegó a 7000 iteraciones de baseline (1.86 M gaussianas, 442 s en la RTX 4060 Laptop),
pero **no hay métricas de test** y el estado quedó en `running`. No usar como resultado.

## 3. Cómo reproducir el 2D

```powershell
.venv\Scripts\python.exe -m papermill --cwd codes codes\01_spc_simulacion_reconstruccion.ipynb salida.ipynb `
  -p DATASET tandt/truck -p NUM_VIEWS 24 -p RUN_NAME paper24
```

Script usado para las cuatro escenas: `outputs/notebooks_ejecutados/run_2d.sh` (log en `run_2d.log`).
Hardware: RTX 4060 Laptop 8 GB, driver 571.96, Windows 11. Código: https://github.com/fralfaro/inf449-proyecto

## 4. TODO

### Paper (2D)
- [x] Agregar la tabla agregada de `paper24` a la sección 2.1 y al paper (hecho 2026-11-07).
- [x] Figuras: calidad frente a razón (agregada), comparación visual y por escena (`latex/informe/figs/`). Faltan mapas de error.
- [x] Completar los `\todo{}` de `conference_101719.tex` (compila a 4 páginas, sin `TODO`).
- [ ] Verificar cada entrada de `references.bib` contra la fuente original.
- [ ] Revisar el límite de páginas del congreso (hoy son 4); hay espacio para figuras por escena (`figs/fig_calidad_<escena>.png`) o mapas de error.
- [x] Compilar, buscar `TODO` en rojo y revisar que no quede ninguno (hecho; revisar el PDF a ojo).

### Repositorio
- [ ] Pasar este archivo al `README.md` (qué se hizo, resultados, TODO, 3D, póster).
- [x] Slides rehechas (`slides/presentacion.qmd`, 23 diapositivas con la portada repetida al final); la de octubre quedó en `presentacion_octubre.qmd`.
- [ ] Mover `codes/3d_gaussian_splatting_low.pdf` fuera de `codes/` o a `docs/`.
- [ ] Commit ordenado: hay borrados y archivos nuevos sin versionar.

## 5. 3D: cómo continuar (para Ayrton)

El notebook `codes/02_gaussian_splatting.ipynb` toma una corrida SPC (`SPC_RUN`) y entrena 3DGS con las imágenes
reconstruidas, con las poses y la nube SfM originales del dataset.

**Pasos recomendados**
1. Validar el baseline: entrenar con las imágenes originales y comparar con el paper en Truck
   (PSNR 23.5 / SSIM 0.840 / LPIPS 0.209 a 7k iteraciones). Si no se acerca, no interpretar el resto.
2. Prueba rápida sobre las 24 vistas de `paper24`: baseline, `control_resample`, `hadamard_cc_r005_tv`,
   `hadamard_cc_r010_tv`, `hadamard_cc_r025_tv`, `hadamard_cc_r050_tv`, a 3000 iteraciones (unos 2–4 min cada una, estimado).
   Se pasa con `PLAN` y `GS_RUN` por papermill:
   ```powershell
   .venv\Scripts\python.exe -m papermill --cwd codes codes\02_gaussian_splatting.ipynb salida_02.ipynb `
     -p SPC_RUN paper24 -p GS_RUN rapida -p ITER_LONG 3000 -p ITER_SHORT 3000 `
     -p PLAN '[["baseline",3000],["hadamard_cc_r025_tv",3000]]'
   ```
3. Si la tendencia es clara, repetir a 7k y luego a 30k con más vistas (por defecto `PLAN = None`).
4. Generar la tabla de propagación 2D → 3D (`propagacion_2d_3d.csv`) y las figuras del notebook.

**Cuidados**
- La GPU es de laptop (8 GB): el notebook espera a que baje de 70 °C entre corridas. Una corrida de 7k con 251 vistas tardó 7.4 min.
- Las poses y la nube SfM vienen del GT: los resultados son una **cota optimista** (ver `00_plan_y_decisiones.md` §7).
- Nunca usar `-r 2` sin consultarlo; cualquier cambio de hiperparámetros debe aplicarse a todas las corridas.
- Si una corrida queda en `running`, volver a ejecutar el notebook con el mismo `GS_RUN`: reanuda desde el checkpoint.

**Extensiones** (ver `docs/03_mejoras_y_trabajo_futuro.md`): re-estimar poses con COLMAP sobre las vistas SPC, entrenar
3DGS directamente contra las mediciones, patrones distintos por vista y muestreo adaptativo.

## 6. Póster (para después)

Mensaje único: *«¿Cuántas mediciones necesita una cámara de un píxel antes de que la reconstrucción se degrade?»*

| Zona | Contenido | Fuente |
|---|---|---|
| Izquierda | Esquema de la SPC y del flujo medición → reconstrucción → 3DGS | `slides/images/spc.png`, `Hadamard.png` |
| Centro | Una vista reconstruida a 5/10/25/50 % (Hadamard + TV) junto al original y al mapa de error | `outputs/spc_runs/.../images` |
| Centro | Gráfico calidad frente a razón, por patrón y método | `summary_2d.csv` |
| Derecha | Si hay 3D: renders de 3DGS por razón y gráfico 2D frente a 3D | `resultados_3dgs.csv` |
| Pie | Conclusión, limitaciones (poses del GT), código y autores | — |

Formato: A0 vertical o el que pida el congreso, plantilla en LaTeX (`tikzposter` o `baposter`) o en PowerPoint.
Reusar las figuras del paper, con tipografía grande y un solo gráfico principal.
