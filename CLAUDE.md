# INF449 – Proyecto final: Single Pixel Camera + 3D Gaussian Splatting

Este archivo indica cómo debe ayudarme Claude en este repositorio. Responder siempre en español neutro y formal.

## Objetivo y alcance

Producto final: un **paper en formato de congreso** (IEEEtran, `latex/informe/conference_101719.tex`) con un nivel de detalle
parecido al informe de la tarea 1 (`latex/ejemplo_informe/informe.tex`): varios ejemplos visuales, tablas por vista y
discusión crítica, pero condensado a formato congreso. El material extra va en anexos o en el repositorio.

**Alcance actual: SOLO 2D.** El paper estudia la simulación de una Single Pixel Camera (SPC) y la reconstrucción de
vistas 2D de escenas de Tanks and Temples / Deep Blending: patrones (Hadamard ordenado, binario aleatorio), ratio de
medición, ruido, reconstrucción directa frente a regularizada (TV-FISTA) y métricas PSNR/SSIM/LPIPS. 3DGS aparece como
**motivación y trabajo futuro**, no como resultado.

**Fuera de alcance por ahora (pendiente para Ayrton Perez):**
- Experimentos 3DGS (Notebook `codes/02_gaussian_splatting.ipynb`): ya funciona de principio a fin, pero solo se
  probó con una versión mínima. Falta ejecutar baseline y corridas SPC por ratio y validar el baseline contra el paper.
- Póster.
- Ver `docs/03_mejoras_y_trabajo_futuro.md` y la sección «Pendientes» de este archivo.

## Autores

- Francisco Alfaro, Departamento de Electrónica, Universidad Técnica Federico Santa María (`francisco.alfaro@usm.cl`)
- Ayrton Perez, Departamento de Informática, Universidad Técnica Federico Santa María (`ayrton.perez@usm.cl`)

Formato de correo: `nombre.apellido@usm.cl`. Ciudad: Valparaíso, Chile.

## Estructura del repositorio

```
codes/        notebooks 01 (SPC 2D) y 02 (3DGS), gs_train_runner.py, README con la instalación
docs/         plan y decisiones, bitácora, trabajo futuro (la fuente de verdad del método)
latex/
  ejemplo_informe/   MODELO de estilo y profundidad (informe de la tarea 1). Solo lectura.
  informe/           el paper: conference_101719.tex + references.bib (IEEEtran)
outputs/      resultados de cada corrida (no versionado)
data/ third_party/   datasets y repo oficial de 3DGS (no versionado)
slides/       presentación Quarto del avance (desactualizada, anterior al rediseño)
```

## Datos técnicos que el paper debe citar

- Código: https://github.com/fralfaro/inf449-proyecto
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GB), driver 571.96. Windows 11.
- Software: Python 3.11, PyTorch 2.11.0+cu128, CUDA Toolkit 12.8, Visual Studio 2022 (MSVC 14.42).
- Transformada rápida de Walsh-Hadamard en GPU: medida en `docs/01_bitacora.md` (3.4 ms por vista de 3 canales a 512×1024).
- Rejilla de medición 512×1024 (N = 524 288); `Phi` no se almacena.

## Fuentes de verdad (usar en este orden)

1. `docs/00_plan_y_decisiones.md`: diseño del experimento y justificación. El paper debe ser consistente con él.
2. `docs/01_bitacora.md`: qué falló y cómo se resolvió; sirve para la sección de limitaciones y reproducibilidad.
3. `codes/01_spc_simulacion_reconstruccion.ipynb` y `outputs/spc_runs/...`: de ahí sale **todo** valor numérico.
4. `latex/ejemplo_informe/informe.tex`: estructura, tono y tipo de figuras y tablas que se esperan.
5. `inf449-portafolio` (carpeta hermana): enunciados, lectures y libro del curso.

## Estructura prevista del paper

Resumen · Introducción · Trabajo relacionado (single pixel, muestreo comprimido, 3DGS) · Modelo de medición SPC ·
Reconstrucción (directa, TV-FISTA) · Diseño experimental (datos, patrones, ruido, métricas) · Resultados (tablas y
figuras por ratio, patrón y método) · Discusión y limitaciones (poses del GT, sRGB frente a radiancia lineal, escena
estática, escena única) · Conclusiones y trabajo futuro (3DGS) · Disponibilidad del código.

## Reglas de trabajo

- **No inventar resultados.** Toda cifra, tabla o figura proviene de una ejecución. Si falta, dejar `\todo{}`/`TODO` y decirlo.
- Las cifras de `outputs/spc_runs/truck/prueba_v32` son de **prueba** (32 vistas, una escena); no van al paper final sin repetir
  con la corrida definitiva.
- No sobrescribir `.tex`, notebooks ni `docs/` sin avisar. No tocar `latex/ejemplo_informe/`, `data/` ni `third_party/`.
- Verificar cada referencia del `.bib` contra la fuente original antes de darla por buena.
- Declarar el uso de herramientas de IA como en el informe de la tarea 1 (sección de material complementario).
- Notebook costoso de ejecutar: avisar antes. El entrenamiento 3DGS ocupa la GPU durante horas.
- Compilar con `latexmk -pdf` dentro de `latex/informe/` y reportar los errores de forma resumida.
- No compilar ni commitear archivos auxiliares de LaTeX (`.aux`, `.log`, `.fdb_latexmk`, `.synctex.gz`).

## Pendientes

2D (tarea actual):
- [ ] Corrida definitiva del Notebook 1: más vistas y más escenas (Truck, Train, DrJohnson, Playroom), semilla fija.
- [ ] Figuras: calidad frente a ratio, comparación visual por patrón y ratio, mapas de error, estudio de ruido y de color.
- [ ] Rellenar el paper con esos resultados y compilar el PDF.
- [ ] Revisar y completar `references.bib`.

3D (para Ayrton, después):
- [ ] Ejecutar `02_gaussian_splatting.ipynb` con una corrida SPC; plan reducido sugerido en `docs/03_mejoras_y_trabajo_futuro.md`.
- [ ] Validar el baseline contra el paper (Truck: PSNR 23.5 / SSIM 0.840 / LPIPS 0.209 a 7k).
- [ ] Tabla de propagación 2D → 3D y póster.
