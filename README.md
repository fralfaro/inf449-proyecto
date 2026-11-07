# Proyecto INF449

Proyecto sobre la simulación de una **Single Pixel Camera (SPC)** y su relación con la reconstrucción 3D de escenas mediante
**3D Gaussian Splatting (3DGS)**.

La pregunta central es cuánto se puede reducir el número de mediciones de la cámara antes de que la calidad caiga, y cómo
se propaga el error de reconstrucción de cada vista a la escena 3D.

## Estado actual

| Parte | Estado |
|---|---|
| Simulación SPC y reconstrucción 2D (Hadamard y aleatorio; directa y TV-FISTA) | **Hecho.** 4 escenas × 24 vistas, razones 5/10/25/50 % |
| Paper IEEE sobre el caso 2D (`latex/informe/`) | **Hecho, en revisión** (5 páginas; faltan verificar referencias y fuentes de figuras) |
| Presentación (`slides/`) | **Hecha, en revisión** (26 diapositivas) |
| Entrenamiento 3DGS con las vistas reconstruidas | **Implementado, sin resultados.** Pendiente |
| Póster | Pendiente |

El paper cubre solo el caso 2D. El flujo 3D está implementado y documentado para continuarlo después.

## Qué se hizo

1. **Entorno local.** PyTorch 2.11 + CUDA 12.8 en Windows, con las extensiones CUDA de 3DGS compiladas
   (receta y errores en `docs/01_bitacora.md`).
2. **Simulación SPC.** Transformada rápida de Walsh-Hadamard en GPU (3.4 ms por vista de 3 canales a 512×1024), con un
   modelo del detector (ruido de Poisson, de lectura y cuantización de 16 bits) y máscaras {0,1} de un DMD.
3. **Reconstrucción.** Directa (mínima norma) y regularizada con variación total (FISTA + FGP).
4. **Experimentos 2D.** Comparación de patrones, razones de medición, método y escena, más estudios de ruido, orden de
   patrones, Bernoulli denso frente a estructurado, y color por canal frente a Bayer.
5. **Documentación y entregables.** Plan y decisiones, bitácora, resultados, paper, slides y mensaje para el equipo.

## Resultados principales (2D)

Media sobre las 4 escenas (Truck, Train, DrJohnson, Playroom), 24 vistas cada una. PSNR en dB; el control de solo
remuestreo da 41.04 dB.

| Patrón · método | 5 % | 10 % | 25 % | 50 % |
|---|---:|---:|---:|---:|
| Hadamard · directa | 26.25 | 28.26 | 31.07 | 32.58 |
| **Hadamard · TV** | **28.05** | **30.11** | **32.95** | **34.63** |
| Aleatorio · directa | 13.62 | 13.93 | 14.94 | 17.10 |
| Aleatorio · TV | 26.52 | 28.56 | 31.37 | 33.44 |

- La TV mejora a la reconstrucción directa en 1.8 a 2.1 dB con Hadamard.
- Con patrones aleatorios la reconstrucción directa no sirve; con TV se acerca a Hadamard a razones altas.
- La escena influye más que el método (≈ 8 dB entre Train y DrJohnson al 5 %). La explicación es una hipótesis sin verificar.

Detalle, tablas por escena y estudios complementarios en [`docs/02_resultados.md`](docs/02_resultados.md).
Tiempos: 12 a 15 minutos por escena, 53 minutos en total, en una RTX 4060 Laptop de 8 GB.

## TODO

### Paper y presentación (2D)
- [ ] Verificar cada entrada de `latex/informe/references.bib` (se escribieron de memoria).
- [ ] Citar la fuente de las figuras con `\todo{}` en el paper y de las marcadas «verificar» en las slides.
- [ ] Leer el PDF del paper y las slides completos, con revisión a ojo del diseño.
- [ ] Revisar el límite de páginas del congreso (hoy son 5) y evaluar mapas de error y figuras por escena.
- [ ] Verificar la hipótesis sobre la diferencia entre escenas (resolución nativa frente a la rejilla).

### 3D (siguiente etapa)
- [ ] Validar el baseline de 3DGS contra el paper original (Truck: PSNR 23.5 / SSIM 0.840 / LPIPS 0.209 a 7k iteraciones).
- [ ] Prueba rápida sobre las 24 vistas de `paper24`: baseline y Hadamard + TV al 5/10/25/50 %, a 3000 iteraciones.
- [ ] Repetir a 7k y 30k con más vistas si la tendencia es clara.
- [ ] Tabla de propagación 2D → 3D y agregar los resultados al paper y a las slides.
- [ ] Re-estimar las poses con COLMAP sobre las vistas SPC (hoy se usan las poses del dataset: cota optimista).
- [ ] Extensiones: entrenar 3DGS directamente contra las mediciones, patrones distintos por vista, muestreo adaptativo.

### Póster
- [ ] Diseñar el póster con un solo mensaje: cuántas mediciones necesita una cámara de un píxel antes de que la
  reconstrucción se degrade. Propuesta en `docs/02_resultados.md`.

### Repositorio
- [ ] Hacer commit y push (hay archivos borrados y nuevos sin versionar).
- [ ] Mover `codes/3d_gaussian_splatting_low.pdf` fuera de `codes/`.

Detalle del 3D y cuidados con la GPU en [`docs/02_resultados.md`](docs/02_resultados.md) y
[`docs/mensaje_para_ayrton.md`](docs/mensaje_para_ayrton.md).

## Estructura del repositorio

```text
.
├── codes/          # Notebooks del pipeline y runner de entrenamiento
│   └── old/        # Intentos previos, solo como referencia
├── docs/           # Plan, bitácora, resultados, trabajo futuro y mensajes
├── latex/
│   ├── ejemplo_informe/   # Informe modelo (tarea 1), solo lectura
│   └── informe/           # Paper IEEE: conference_101719.tex y references.bib
├── slides/         # Presentación escrita en Quarto
├── data/           # Datasets (no versionado)
├── outputs/        # Resultados de cada corrida (no versionado)
├── third_party/    # Repo oficial de 3DGS (no versionado)
├── CLAUDE.md       # Alcance y reglas de trabajo con Claude
├── requirements.txt
├── LICENSE
└── README.md
```

## Ejecución

El proyecto corre en local, en Windows, con una GPU NVIDIA. La instalación del entorno, la compilación de las extensiones
CUDA y la ejecución están en [`codes/README.md`](codes/README.md). Una escena del paso 1 y 2 toma de 12 a 15 minutos:

```powershell
.venv\Scripts\python.exe -m papermill --cwd codes codes\01_spc_simulacion_reconstruccion.ipynb salida.ipynb `
  -p DATASET tandt/truck -p NUM_VIEWS 24 -p RUN_NAME paper24
```

| Paso | Notebook |
|---|---|
| 1 y 2. Simular la SPC y reconstruir las vistas | [`codes/01_spc_simulacion_reconstruccion.ipynb`](codes/01_spc_simulacion_reconstruccion.ipynb) |
| 3 y 4. Preparar escenas, entrenar 3DGS y comparar | [`codes/02_gaussian_splatting.ipynb`](codes/02_gaussian_splatting.ipynb) |

## Documentación

- [`docs/00_plan_y_decisiones.md`](docs/00_plan_y_decisiones.md): decisiones de diseño y su justificación.
- [`docs/01_bitacora.md`](docs/01_bitacora.md): qué se hizo, qué falló y cómo se resolvió.
- [`docs/02_resultados.md`](docs/02_resultados.md): tablas, figuras, interpretación, pendientes y guía para el 3D y el póster.
- [`docs/03_mejoras_y_trabajo_futuro.md`](docs/03_mejoras_y_trabajo_futuro.md): mejoras posibles.
- [`docs/mensaje_para_ayrton.md`](docs/mensaje_para_ayrton.md): mensaje de traspaso para el equipo.
- [`docs/prompt_opus.md`](docs/prompt_opus.md): prompt para mejorar el paper y las slides con un modelo más capaz.

## Paper y presentación

- Paper: [`latex/informe/conference_101719.tex`](latex/informe/conference_101719.tex). Compilar con `latexmk -pdf` dentro de `latex/informe/`.
- Presentación: [`slides/presentacion.qmd`](slides/presentacion.qmd), con Quarto (`quarto render presentacion.qmd`). Se genera
  `presentacion.html`; no hay PDF.

## Integrantes

- Francisco Alfaro, Departamento de Electrónica, Universidad Técnica Federico Santa María
- Ayrton Perez, Departamento de Informática, Universidad Técnica Federico Santa María
