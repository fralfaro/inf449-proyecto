# Proyecto INF449

Proyecto sobre la simulación de una **Single Pixel Camera (SPC)** y la reconstrucción 3D de escenas mediante **Gaussian Splatting**.

El flujo general es:

1. Simular las mediciones de una Single Pixel Camera.
2. Reconstruir las vistas de la escena.
3. Preparar los datos y las poses para la reconstrucción 3D.
4. Entrenar Gaussian Splatting y comparar los resultados.

## Estructura del repositorio

```text
.
├── codes/          # Códigos y notebooks para ejecutar en Google Colab
├── slides/         # Presentación escrita en Quarto
├── LICENSE
└── README.md
```

Los notebooks principales se encuentran en [`codes/`](codes/). La presentación y sus archivos de estilo están en [`slides/`](slides/).

## Configuración de Google Colab

Para ejecutar los notebooks con GPU:

1. Abrir el archivo `.ipynb` desde la carpeta [`codes/`](codes/) en Google Colab.
2. Ir a **Entorno de ejecución → Cambiar tipo de entorno de ejecución**.
3. En **Acelerador de hardware**, seleccionar **T4 GPU**.
4. Presionar **Guardar** y ejecutar las celdas en orden.
5. Verificar la GPU ejecutando:

   ```python
   !nvidia-smi
   ```

6. Cuando el notebook lo solicite, permitir el acceso a Google Drive.

Se recomienda comenzar con los valores de prueba indicados dentro de cada notebook antes de ejecutar un entrenamiento completo. Los resultados persistentes deben guardarse en Google Drive para no perderlos al reiniciar la sesión de Colab.

## Indicaciones de ejecución

- Ejecutar primero `01_single_pixel_camera_colab.ipynb` para generar las vistas y los datos de entrada.
- Ejecutar después `02_gaussian_splatting_colab.ipynb` para entrenar y evaluar Gaussian Splatting.
- Mantener las rutas de Google Drive consistentes entre ambos notebooks.
- Revisar que la sesión de Colab mantenga activa la GPU T4 antes de comenzar un entrenamiento largo.
- Los notebooks adicionales de la carpeta `codes/` sirven como material de apoyo, diagnóstico o referencia.

## Presentación

La presentación está en [`slides/presentacion.qmd`](slides/presentacion.qmd) y utiliza Quarto. El PDF generado se encuentra en [`slides/presentacion.pdf`](slides/presentacion.pdf).

## Integrantes

- Francisco Alfaro
- Ayrton Perez
