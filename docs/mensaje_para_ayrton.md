# Mensaje para Ayrton

> Texto listo para copiar y pegar. Antes de enviarlo, Francisco debe hacer el commit y el push (hoy todo está sin versionar).

---

Hola Ayrton,

Dejé avanzada la parte 2D del proyecto y te paso lo que necesito que revises y lo que queda para ti. Todo está en
https://github.com/fralfaro/inf449-proyecto (rama `main`). Lee primero `CLAUDE.md` y `docs/02_resultados.md`: ahí está el
alcance, lo hecho y los pendientes.

**Qué hay hecho**
- Simulación SPC 2D y reconstrucción (Hadamard ordenado y aleatorio; directa y TV-FISTA) en
  `codes/01_spc_simulacion_reconstruccion.ipynb`.
- Corrida definitiva `paper24`: 4 escenas × 24 vistas, razones 5/10/25/50 %. Duró 53 min en mi RTX 4060 Laptop.
  Resultados en `outputs/spc_runs/<escena>/paper24/` y agregados en `outputs/spc_runs/resumen_paper24_*.csv`.
- Paper en formato IEEE (en español, 4 páginas): `latex/informe/conference_101719.tex` y `references.bib`.
- Slides (23 diapositivas, la última repite la portada): `slides/presentacion.qmd`.
- Notebook 3DGS (`codes/02_gaussian_splatting.ipynb` + `codes/gs_train_runner.py`): funciona de punta a punta, pero **solo
  se probó con una versión mínima**. No hay resultados 3D.

**Qué te pido revisar** (en este orden)
1. **Las cifras del paper contra los CSV.** Cada número sale de `outputs/spc_runs/resumen_paper24_todas_escenas.csv`
   o de `estudio_*.csv`. Si algo no coincide, avísame.
2. **`latex/informe/references.bib`.** Escribí las referencias de memoria: hay que verificar autores, año, revista y
   páginas de cada una antes de enviar.
3. **El método** (`docs/00_plan_y_decisiones.md`): ¿te parecen razonables la rejilla 512×1024, el modelo del detector y
   la elección de λ en vistas de entrenamiento?
4. **El código del Notebook 1**, en especial la métrica (siempre contra la imagen original) y la semilla.
5. **Las slides**: la figura del pipeline de 3DGS salió del material de ejemplo y su fuente no está verificada.
6. **El PDF del paper**: compílalo (`latexmk -pdf` dentro de `latex/informe/`) y léelo completo. Yo solo revisé que
   compile sin referencias rotas.

**Cómo replicar el 2D** (una escena, unos 12 a 15 min)
```powershell
.venv\Scripts\python.exe -m papermill --cwd codes codes\01_spc_simulacion_reconstruccion.ipynb salida.ipynb `
  -p DATASET tandt/truck -p NUM_VIEWS 24 -p RUN_NAME paper24
```
Necesitas el entorno de `codes/README.md` (Python 3.11, PyTorch 2.11 + CUDA 12.8, las dos extensiones CUDA compiladas),
los datos en `data/tandt_db/` y el repo oficial de 3DGS en `third_party/gaussian-splatting/`. `data/`, `outputs/` y
`third_party/` no están en git. Un aviso `Passed unknown parameter: NUM_VIEWS` de papermill es inofensivo.

**Lo que queda para ti**
1. **Experimentos 3D.** Detalle en `docs/02_resultados.md` §5. Resumen:
   - Valida primero el baseline contra el paper de 3DGS (Truck: PSNR 23.5 / SSIM 0.840 / LPIPS 0.209 a 7k iteraciones).
     Si no se acerca, no interpretes las corridas SPC.
   - Prueba rápida sobre las 24 vistas de `paper24`: baseline y Hadamard + TV al 5/10/25/50 %, a 3000 iteraciones
     (calculo 2–4 min por corrida; es una estimación, no la medí).
   - Cuida la GPU de laptop: el notebook espera a que baje de 70 °C entre corridas.
   - La corrida vieja `outputs/gs_runs/truck/prueba_v251` **no sirve**: no tiene métricas de test y quedó en `running`.
   - Las poses y la nube SfM vienen del GT: los resultados serán una cota optimista (está declarado en el paper).
2. **Póster.** Propuesta en `docs/02_resultados.md` §6: mensaje único («¿cuántas mediciones necesita una cámara de un píxel
   antes de que la reconstrucción se degrade?»), figuras del paper y, si hay 3D, renders por razón.
3. **Cuando haya resultados 3D**, agregarlos al paper (hoy están como trabajo futuro) y a las slides.

Cualquier duda me escribes. Gracias.

Francisco
