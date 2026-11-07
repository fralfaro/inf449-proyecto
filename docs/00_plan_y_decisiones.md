# Plan y decisiones de diseño

Estado de cada decisión: **[verificada]** (comprobada ejecutando o leyendo el código) o
**[propuesta]** (aún por validar con la corrida de prueba). Las cifras de resultados van en
`02_resultados.md` y solo se escriben después de ejecutar.

## 1. Entorno

| Elemento | Estado encontrado | Consecuencia |
|---|---|---|
| `.venv` (Python 3.11.14, uv) | `torch 2.14.1+cpu`; sin CUDA, `plyfile` ni papermill | Hay que instalar torch con CUDA |
| Python del sistema 3.12.4 | `torch 2.11.0+cu128` con CUDA; sin matplotlib, scikit-image ni lpips | Solo se usó para un benchmark |
| GPU | RTX 4060 Laptop, 8 GB, driver 571.96 | Limita los hiperparámetros de 3DGS |
| Compiladores | VS 2022 (MSVC 14.42), CUDA Toolkit 12.8 (`nvcc`), Windows SDK 10.0.22000 | Se pueden compilar las extensiones |
| `diff_gaussian_rasterization`, `simple_knn` | **No compiladas** (sin `.pyd`), no importan | Hay que compilarlas |
| COLMAP | No instalado | La ablación de re-estimar poses queda fuera salvo que se pida |
| Pesos LPIPS | Sin caché local | Hay que descargarlos (AlexNet 244 MB, VGG16 553 MB) |

**[verificada] Se usa `.venv`** con `torch 2.11.0+cu128`, la versión más nueva publicada para
CUDA 12.8 y Python 3.11. Coincide con el `nvcc` 12.8 instalado y no modifica el Python del sistema.
Las extensiones compilaron sin editar sus fuentes; la receta está en `codes/README.md` y los errores en
`01_bitacora.md`. La tabla anterior describe el estado encontrado, antes de instalar.

## 2. Datos

Inspección de `cameras.bin`, `images.bin` y `points3D.bin` **[verificada]**:

| Escena | Imágenes = poses | Tamaño de archivo | `cameras.bin` (modelo, W×H) | Puntos 3D |
|---|---|---|---|---|
| tandt/truck | 251 | 979×546 | PINHOLE, 1957×1091 | 136 029 |
| tandt/train | 301 | 980×545 | PINHOLE, 1959×1090 | 182 686 |
| db/drjohnson | 263 | 1332×876 | PINHOLE, 1332×876 | 80 861 |
| db/playroom | 225 | 1264×832 | PINHOLE, 1264×832 | 37 005 |

En las cuatro escenas los nombres de las imágenes coinciden exactamente con los de `images.bin`.

**Los intrínsecos no se reescriben [verificada leyendo el código].** En Truck y Train,
`cameras.bin` ya declara el doble de la resolución de los archivos.
`scene/dataset_readers.py` calcula el FoV con el foco y el ancho de `cameras.bin`, y nunca con
el tamaño del archivo. Por eso basta con conservar la razón de aspecto, y era innecesario lo que
hacían los notebooks antiguos: volver a escalar cada vista a la resolución original "para no
invalidar las poses".

## 3. Lecciones de los intentos previos (`codes/old/`, `outputs/`)

| Observación | Causa | Decisión |
|---|---|---|
| PSNR ~17–20 dB y SSIM ~0.43–0.49 para cualquier ratio | Se simulaba a 64×64 y se escalaba a 979×546; domina el cuello de botella de resolución y no el ratio | Simular cerca de la resolución nativa con transformada rápida |
| Distorsión de aspecto | 979×546 se llevaba a un cuadrado de 64×64 | Rejilla rectangular 2^a×2^b |
| `spc_local`: imágenes `01.jpg`, `04.jpg`… con un `images.bin` del mismo tamaño que el original (nombres `000001.jpg`) | Renombrado de vistas; el lector abre `basename(extr.name)`, así que no las encontraría | Conservar el nombre de pose y filtrar `images.bin` al usar un subconjunto |
| `gs_outputs/` y `logs/` vacíos; `main.ipynb` termina en `VALIDACIÓN 3D FALLIDA` y la conclusión es una plantilla | 3DGS nunca terminó en local. La celda 42 falló con `maximum recursion depth exceeded`, y `subprocess.run` estaba envuelto para medir tiempos | Medir tiempos con un runner propio, sin parchear `subprocess` |
| Patrones aleatorios con ~6 dB | Retroproyección como "inversa" de una matriz no ortogonal | El aleatorio se reconstruye con un método regularizado |
| PSNR 2D no comparable entre tablas | Se comparaba a veces a 64×64 y a veces a resolución original | Una sola función de métricas, siempre a la resolución final |
| `metrics.py` compara contra el GT de la carpeta de entrada | En una escena SPC ese GT ya está degradado | Métrica principal contra las imágenes **originales** |

Lo que sí funcionó y se reutiliza: el flujo de compilación de `gaussian_splatting_colab.ipynb`
(parches de cabeceras, `--no-build-isolation`) y la selección de vistas por nombre de pose.

## 4. Simulación SPC [propuesta]

- **Rejilla de medición:** `H×W = 512×1024` (N = 524 288), casi igual en píxeles a 979×546.
  Hadamard exige potencias de 2; el cambio de aspecto equivale a macro-píxeles no cuadrados del
  DMD y se deshace al guardar. Una condición de control (solo remuestreo, sin SPC) aísla esa pérdida.
- **Matriz libre:** `Phi` no se almacena (serían 524 288² entradas). Se usa una transformada rápida
  de Walsh-Hadamard en GPU; basta guardar índices o semilla.
  Benchmark real: 3.4 ms/vista (3 canales) a 512×1024, con 0.48 GiB para un lote de 16.
- **Patrones:**
  - Hadamard ordenado. El orden *cake-cutting* tiene forma analítica porque un patrón separable
    `w_i(x)·w_j(y)` tiene exactamente `(s_i+1)(s_j+1)` bloques conexos. También *Russian dolls*.
  - Binario aleatorio mediante matriz estructuralmente aleatoria (permutación de píxeles +
    Hadamard + submuestreo). Un Bernoulli i.i.d. denso no cabe a esta resolución (1.7 GB a 5 % y
    17 GB a 50 %, a 1 bit por entrada), así que se valida contra Bernoulli denso a baja resolución.
  - Fourier opcional, de 4 pasos, contando las 4 mediciones por coeficiente.
- **Modelo físico:** un DMD solo tiene máscaras {0,1}, así que se simula
  `y = <P,x>` con `P = (1+h)/2`, más una medición DC. Encima se aplica Poisson + gaussiano +
  ADC de n bits. El coeficiente en ±1 se recupera con `c = 2y − DC`, lo que amplifica el ruido
  por 2. Este efecto suele omitirse y se mantiene explícito.
- **Color:** por canal (3M mediciones) frente a máscara Bayer (M mediciones, TV conjunta).
  La comparación se hace en 2D reportando el costo total de mediciones.

## 5. Reconstrucción [propuesta]

- **(a) Directa:** transpuesta con ceros en los coeficientes no medidos. Es la solución de mínima
  norma, porque `Phi Phiᵀ = I`.
- **(b) Regularizada:** TV con FISTA (Beck & Teboulle 2009) y prox TV por FGP, en lotes en GPU.
- λ se elige con una búsqueda en un subconjunto de vistas **de entrenamiento** y se fija antes de
  medir las vistas de test, para no sesgar la evaluación.

## 6. 3DGS [propuesta]

- **Poses:** se usan `cameras`, `images` y `points3D` originales, porque el experimento aísla el
  efecto de la adquisición. Ver la limitación en la sección 7.
- **Comparabilidad:** mismas iteraciones e hiperparámetros, y la semilla 0 de `safe_state`, con
  `--eval` y el test LLFF 1/8 del repo oficial. Las vistas de test se evalúan contra el original.
- **Validación del baseline:** el paper reporta para Truck PSNR 23.506 / SSIM 0.840 / LPIPS 0.209 a
  7k y 25.187 / 0.879 / 0.148 a 30k (Tablas 7–9). El baseline debe acercarse a estos valores antes
  de interpretar las corridas SPC. El paper usa una GPU A6000, así que no se comparan tiempos.
- **Pérdida y densificación:** sin cambios respecto del paper salvo necesidad de VRAM: L1 + D-SSIM
  con λ=0.2 (§5.1, Ec. 7), densificación cada 100 iteraciones entre 500 y 15k (§5.2), SH grado 3.
- **VRAM:** primero `data_device=cuda` a resolución nativa. Si hay OOM, en este orden:
  `data_device=cpu`, `densify_grad_threshold` mayor, y como último recurso `-r 2`. Cualquier ajuste
  se aplica a todas las corridas.
- **JPEG:** los nombres de pose terminan en `.jpg`, así que las vistas se guardan en JPEG con
  calidad 100 y sin submuestreo de croma. Se mide el error que introduce.
- **Resolución:** nunca se usa `-r 2` sin consultarlo antes (condición del usuario).
- **Corridas:** en secuencia, con resultados guardados al terminar cada una, reanudación desde checkpoint y
  registro de temperatura. Si el tiempo aprieta, se prioriza baseline y Hadamard + TV a 7k y 30k.
- **Control de interpretación:** si el baseline en Truck no se acerca al paper, las corridas SPC no se
  interpretan hasta aclararlo.

## 7. Limitación principal: poses y nube SfM del GT

**Todas las corridas, incluidas las SPC, usan las poses y la nube SfM calculadas por COLMAP sobre las imágenes
originales de alta calidad.** Una SPC real no dispondría de ellas: tendría que estimarlas desde vistas
reconstruidas y degradadas, o medirlas con otro sensor.

| Qué se filtra del GT | Efecto probable en los resultados |
|---|---|
| Poses de cámara | Se elimina el error de calibración, que crecería a ratios bajos. COLMAP podría incluso no registrar vistas muy degradadas |
| Nube SfM inicial (posición y color) | Las gaussianas parten de geometría y color correctos; la degradación solo afecta a la optimización |

**Consecuencia.** Los resultados miden el efecto de la **calidad fotométrica** de la adquisición, con geometría
inicial ideal. Son una **cota optimista** del sistema completo: el ratio mínimo aceptable que se reporte vale para
este escenario, y un sistema real probablemente necesitaría más mediciones.

**Por qué se acepta.** Aísla una sola variable, permite medir todas las corridas contra las mismas cámaras, y
COLMAP no está instalado. El paper respalda que la inicialización importa (§7.3, Fig. 7): con puntos aleatorios
la calidad cae, sobre todo en el fondo. La re-estimación de poses queda en `03_mejoras_y_trabajo_futuro.md`.

Otras simplificaciones de la simulación, de menor peso:

- Las vistas son imágenes ya capturadas, no una captura óptica. Las mediciones heredan el JPEG y el desenfoque del dataset.
- Se mide en valores sRGB, no en radiancia lineal. Se cuantifica en la sección 9.3 del Notebook 1.
- La escena se asume estática durante las mediciones de cada vista.
