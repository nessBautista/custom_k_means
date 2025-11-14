# Custom K-Means Clustering

## Instalación

Este proyecto utiliza [uv](https://docs.astral.sh/uv/) como gestor de paquetes y entornos virtuales de Python.

### Instalar uv

```bash
# macOS y Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Para más opciones de instalación, consulta la [documentación oficial de uv](https://docs.astral.sh/uv/getting-started/installation/).

### Instalar dependencias del proyecto

```bash
# Instalar todas las dependencias definidas en pyproject.toml
uv sync

# Activar el entorno virtual creado por uv
source .venv/bin/activate  # macOS/Linux
# o
.venv\Scripts\activate  # Windows
```

# Modulos

## Funciones Utilitarias

### Módulo Data Loader

El módulo `src/data_loader` proporciona una interfaz unificada para acceder a los datos del proyecto. Permite cargar imágenes del dataset BSD500 como numpy arrays, así como archivos JSON con snapshots de parámetros y resultados de experimentos previos. Los JSON snapshots son esenciales porque contienen los mejores valores encontrados para los parámetros (como k_clusters, velocity_method, alpha, etc.) en experimentos anteriores, junto con las métricas de calidad obtenidas (PRI scores, mejoras, conteos de contornos). El DataLoader carga automáticamente la versión más reciente cuando existen múltiples archivos para la misma imagen, y localiza el directorio de datos sin importar desde dónde se ejecute el código.

```mermaid
graph LR
    DL[Data Loader] --> IMG[Carga de imágenes]
    DL --> JSON[Carga de JSON con mejores parámetros]
```

### Módulo de Visualización

El módulo `src/viz` concentra toda la lógica de visualización para evitar cargar el Marimo Notebook con métodos de creación de plots y gráficos. Proporciona funciones especializadas para mostrar imágenes del dataset BSD500 en grids horizontales, así como para visualizar resultados de K-Means clustering en layouts estructurados. Las funciones manejan automáticamente la configuración de subplots, títulos, ejes y formatos de imagen, permitiendo que el notebook se mantenga limpio con llamadas simples de una línea que retornan figuras listas para mostrar.

```mermaid
graph LR
    VIZ[Visualización] --> BSD[Display de imágenes BSD500]
    VIZ --> KMEANS[Display de resultados K-Means]
```

### Módulo K-Means Clustering

El módulo `src/kmeans` proporciona un wrapper limpio sobre [sklearn.cluster.KMeans](https://scikit-learn.org/0.19/modules/generated/sklearn.cluster.KMeans.html) específicamente diseñado para segmentación de imágenes. Simplifica el flujo de trabajo al manejar automáticamente el reshaping de imágenes RGB, normalización de datos, y generación de imágenes segmentadas con colores cuantizados. El módulo expone tres componentes principales: `KMeansConfig` para configuración de parámetros (número de clusters, iteraciones, método de inicialización), la clase `KMeans` que encapsula el algoritmo de sklearn, y `process_images_batch()` para procesamiento eficiente de múltiples imágenes usando parámetros guardados en snapshots.

```mermaid
graph LR
    KMEANS --> FIT[fit_image]
    FIT --> SEGMENT[get_segmented_image]
    KMEANS --> BATCH[process_images_batch]
```

### Módulo Level Set

El módulo `src/level_set` refina las segmentaciones de K-Means mediante level set evolution, suavizando fronteras mientras las detiene en bordes fuertes de la imagen. Utiliza [scikit-fmm](https://github.com/scikit-fmm/scikit-fmm) para calcular signed distance functions mediante Fast Marching Method, transformando las etiquetas discretas de K-Means en representaciones continuas donde cada píxel almacena su distancia al borde del cluster más cercano. Para el cálculo de velocidad, emplea [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html) que proporciona herramientas fundamentales para implementar velocidades personalizadas de level set: cálculo de gradientes mediante filtros Sobel, estimación de curvatura por diferencias finitas, y funciones edge-stopping basadas en gradiente inverso gaussiano. El módulo ejecuta evolución iterativa con reinicialización periódica para mantener estabilidad numérica.

```mermaid
graph LR
    LevelSet --> INIT[process_levelsets_batch]
    LevelSet --> EVOLVE[process_evolution_batch]
    LevelSet --> EXTRACT[extract_evolved_labels_batch]
```