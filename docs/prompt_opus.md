# Prompt para mejorar el paper y la presentación

Uso: abrir Claude Code en la raíz del repositorio con un modelo más capaz (por ejemplo Opus), pegar el prompt y dejarlo
trabajar. Está escrito para que el modelo parta del contexto del repo y no invente cifras.

---

## Prompt

```text
Contexto. Trabaja en el repositorio de mi proyecto de INF449 (Fotografía Computacional, UTFSM). Lee primero, en este
orden: CLAUDE.md, README.md, docs/00_plan_y_decisiones.md, docs/02_resultados.md y docs/01_bitacora.md. El proyecto
simula una Single Pixel Camera (SPC) sobre vistas de escenas 3D y compara patrones (Hadamard ordenado, binario
aleatorio) y reconstrucción (directa y TV-FISTA). El alcance del paper es SOLO 2D; 3D Gaussian Splatting es motivación y
trabajo futuro.

Archivos que debes mejorar:
- Paper IEEE: latex/informe/conference_101719.tex y latex/informe/references.bib
- Presentación: slides/presentacion.qmd (estilo en slides/style.scss; el modelo de estilo está en slides/slide_ejemplos/)
Fuentes de verdad numéricas, y únicas permitidas para cualquier cifra:
- outputs/spc_runs/resumen_paper24_todas_escenas.csv y resumen_paper24_media_escenas.csv
- outputs/spc_runs/<escena>/paper24/summary_2d.csv, estudio_ruido.csv, estudio_patrones.csv, estudio_color.csv,
  estudio_bernoulli.csv, lambda_search.csv
Los estudios complementarios usan solo 3 vistas de Truck: trátalos como indicativos, nunca como evidencia robusta.

Reglas (obligatorias).
1. No inventes ni redondees cifras por tu cuenta. Cada número del texto debe poder rastrearse a un CSV; al terminar,
   entrégame una tabla «cifra del texto → archivo y fila». Si una cifra no está en los CSV, no la uses.
2. No afirmes causas que no se midieron. La explicación de la diferencia entre escenas (resolución nativa frente a la
   rejilla 512x1024) es una hipótesis; si quieres verificarla, propón un experimento barato y dime cuál es, pero no lo ejecutes
   sin avisarme (ocupa la GPU).
3. No ejecutes entrenamientos de 3DGS ni el Notebook 1 completo. Sí puedes ejecutar los scripts de figuras en
   outputs/notebooks_ejecutados/ y compilar el LaTeX y Quarto.
4. No toques latex/ejemplo_informe/, data/, third_party/ ni los notebooks. No hagas commits.
5. Las referencias del .bib se escribieron de memoria: verifícalas con búsqueda web contra la fuente original (autores,
   año, revista, volumen, páginas, DOI) y corrige o elimina las que no puedas confirmar. Lista los cambios.
6. Las figuras marcadas con \todo{} o «verificar» (single-pixel.png, Hadamard.png, 3dgs.jpeg, hadamard.jpg) no tienen fuente.
   Intenta identificarla con búsqueda inversa o por su texto; si no la encuentras, propón reemplazarla por un diagrama propio
   y dime qué dibujarías. No cites una fuente que no hayas confirmado.
7. Escribe en español neutro y formal. El paper debe sonar a artículo de congreso, no a informe de tarea.

Tareas sobre el PAPER.
a) Lee el PDF compilado completo (latexmk -pdf en latex/informe/) y revisa el diseño: tamaño y posición de figuras, tablas,
   desbordes, espacios. Corrígelo.
b) Mejora el resumen, la introducción y la motivación: el problema, la brecha, la pregunta, las contribuciones y el alcance
   deben quedar claros en la primera página. El resumen no debe tener fórmulas ni símbolos especiales.
c) Refuerza el trabajo relacionado con estado del arte real de SPC (muestreo comprimido, órdenes de patrones, reconstrucción
   con redes) y de 3DGS, citando solo lo que verifiques.
d) Revisa la sección de resultados: que cada afirmación esté respaldada por una cifra o figura, que se discutan las
   limitaciones de n (24 vistas por escena, 4 escenas, 3 vistas en los estudios) y que no haya sobreinterpretación.
   Añade una comparación explícita con la literatura cuando sea posible.
e) Propón (y, si es barato, genera con los scripts existentes) figuras que faltan: mapas de error, curvas por escena,
   comparación visual en otra escena. Respeta que el paper tenga como máximo 6 páginas.
f) Elimina los \todo{} que logres resolver; deja los que no, y listalos.

Tareas sobre la PRESENTACIÓN.
a) Renderiza con quarto render y revisa cada diapositiva como imagen (usa Chrome headless; si sale en blanco, borra
   slides/presentacion_files y renderiza de nuevo). Corrige desbordes, espacios vacíos y legibilidad.
b) La charla debe durar 12 a 15 minutos: revisa el ritmo, que haya una idea por diapositiva y notas del orador útiles.
c) Debe mantenerse: la última diapositiva repite la portada, el logo es slides/images/Logo_UTFSM.png y hay pocas
   referencias (hoy 10).
d) Mejora las explicaciones de SPC, Hadamard, TV-FISTA y 3DGS para una audiencia de posgrado en imágenes computacionales,
   con las ecuaciones correctas y coherentes con el paper.
e) Añade una diapositiva de preguntas probables con respuestas basadas en los resultados.

Entrega.
1. Cambios aplicados en los archivos (no copies el contenido en el chat).
2. Tabla de trazabilidad de cifras.
3. Lista de referencias y figuras verificadas, corregidas o pendientes.
4. Lista de lo que no pudiste resolver y lo que necesito decidir yo.
5. Actualiza docs/02_resultados.md y la sección TODO del README.md con el estado final.

Antes de empezar, dame en 10 líneas tu plan y qué vas a verificar primero.
```

---

## Notas para quien lo use

- Si el modelo propone ejecutar experimentos adicionales (por ejemplo verificar la hipótesis de resolución nativa), decide
  si vale la pena: usan la GPU del laptop.
- Pídele que trabaje por partes si la sesión se hace larga: primero el paper, luego la presentación.
- Revisa a ojo el PDF y las slides al final: las capturas automáticas pueden salir en blanco.
