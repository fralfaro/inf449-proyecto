# Mejoras y trabajo futuro

Ideas ordenadas de mayor a menor relevancia para la pregunta del proyecto. Ninguna está implementada ni
evaluada, salvo donde se indica.

## Cerrar la brecha con una SPC real

- **Estimar las poses sin el GT.** El punto más débil del experimento: las poses y la nube inicial provienen
  de COLMAP sobre las imágenes originales. Un paso natural es re-estimarlas con COLMAP sobre las vistas SPC y
  comparar con las del dataset. Si fallan a ratios bajos, se pueden probar métodos que optimizan poses y
  gaussianas en conjunto. Requiere instalar COLMAP.
- **Inicialización sin SfM.** Medir el costo de partir de puntos aleatorios, en la línea de la ablación del
  paper (§7.3, Fig. 7). El repo solo lo hace para escenas sintéticas, así que necesita un pequeño cambio.
- **Luz lineal.** La simulación principal mide valores sRGB, igual que gran parte de la literatura de
  simulación SPC, mientras que un fotodiodo integra radiancia lineal. El Notebook 1 incluye un estudio
  acotado; falta repetir el experimento completo.
- **Movimiento.** Se asume una escena estática durante todas las mediciones de una vista. Una captura real
  tarda $M$ / (tasa del DMD) por vista, lo que limita las escenas posibles.

## Entrenar 3DGS desde las mediciones

Hoy el pipeline reconstruye cada vista y entrena con esas imágenes, de modo que 3DGS hereda los artefactos
de la reconstrucción. La alternativa es saltarse la reconstrucción y optimizar las gaussianas directamente
contra las mediciones:

$$\min_{\theta}\ \sum_{v} \big\|\Phi_v\,\mathcal R_v(\theta) - y_v\big\|_2^2$$

donde $\mathcal R_v$ es el render de la vista $v$. Con el operador rápido de Hadamard el costo por iteración
es bajo, y la geometría 3D compartida actuaría como prior, sobre todo con patrones distintos en cada vista.
Es la extensión más prometedora.

## Muestreo

- **Patrones multivista.** Hoy los patrones ordenados son iguales en todas las vistas, de modo que ninguna
  mide altas frecuencias. Variarlos entre vistas haría complementaria la información.
- **Muestreo adaptativo.** Elegir los patrones siguientes según lo ya medido, o según el error del render 3D
  actual, concentrando mediciones donde más se necesitan.
- **Reparto de mediciones por vista y por canal.** Dedicar más mediciones a la luminancia que a la crominancia,
  o a las vistas que más aportan a la geometría.

## Reconstrucción

- **Priors aprendidos.** Plug-and-play con un denoiser, o una red entrenada sobre pares medición-imagen, en vez de TV.
- **Priors entre vistas.** Reconstruir cada vista aprovechando sus vecinas ya alineadas con las poses.

## Evaluación

- Repetir con varias semillas para estimar la variabilidad entre corridas.
- Extender a las otras escenas disponibles (`train`, `drjohnson`, `playroom`), incluidas las interiores.
- Medir la geometría, no solo el aspecto de los renders. Por ejemplo, la distribución de las gaussianas y
  la presencia de *floaters* a ratios bajos.
