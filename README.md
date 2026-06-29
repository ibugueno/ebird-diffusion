# Ebird Diffusion

Reconstrucción de imágenes a partir de eventos mediante un modelo de difusión
DDPM condicionado. El proyecto entrena primero una U-Net con imágenes MNIST y,
posteriormente, una rama condicional con representaciones de eventos N-MNIST.

Esta rama deriva del trabajo de Fabián Valderrama disponible en `Ebird_MNIST`.
La rama `fv-mnist` conserva esa versión sin modificaciones; `develop` contiene
los ajustes de Docker, portabilidad y validación del entorno.

## Flujo del modelo

El proceso completo consta de tres etapas:

1. Entrenamiento de la U-Net DDPM con imágenes MNIST.
2. Entrenamiento de la rama condicional con pares MNIST/N-MNIST.
3. Generación de imágenes y evaluación de los checkpoints.

El orquestador de las tres etapas es `src/run.py`.

## Requisitos del servidor

- Docker Engine.
- GPU NVIDIA.
- Driver NVIDIA compatible con CUDA 12.1.
- NVIDIA Container Toolkit.
- Espacio para datasets, checkpoints y muestras generadas.

La imagen utiliza Ubuntu 20.04, Python 3.10, Miniconda, PyTorch 2.1 y CUDA 12.1.

> El código actual selecciona `cuda:0` y no implementa DDP ni DataParallel.
> Aunque `run_docker.sh` expone varias GPU, un entrenamiento individual utiliza
> solamente la primera GPU visible.

## Construir la imagen

Desde la raíz del repositorio:

```bash
docker build -t ignacio_event_ebird .
```

El Dockerfile abre Bash por defecto; no inicia automáticamente un entrenamiento.

## Estructura esperada del dataset

La ruta de entrada configurada en `run_docker.sh` debe contener directamente:

```text
reconstruction/
├── MNIST/
│   ├── Train/
│   │   ├── 0/
│   │   └── ... 9/
│   └── Test/
│       ├── 0/
│       └── ... 9/
└── N-MNIST/
    └── 33ms/
        ├── Train/
        │   ├── 0/
        │   └── ... 9/
        └── Test/
            ├── 0/
            └── ... 9/
```

Las imágenes MNIST y N-MNIST emparejadas deben compartir el mismo nombre de
archivo dentro de cada clase.

## Levantar el contenedor

El script incluido configura GPU, memoria compartida y montajes:

```bash
./run_docker.sh
```

Los volúmenes quedan organizados así:

| Host | Contenedor | Uso |
|---|---|---|
| `/home/ignacio.bugueno/cachefs/datasets/processed_data/reconstruction` | `/app/Rislab_Event_influence_volume/dataset` | Dataset de entrada, sólo lectura |
| `/home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird` | `/app/Rislab_Event_influence_volume` | Checkpoints, logs, muestras y reportes |

Para otro servidor, edita únicamente las rutas del lado izquierdo en
`run_docker.sh`.

## Smoke test sin dataset

Una vez dentro del contenedor, se puede validar el entorno sin disponer todavía
de MNIST/N-MNIST:

```bash
python tests/smoke_test.py --device cuda
```

La prueba comprueba:

- disponibilidad de PyTorch y CUDA;
- carga de imágenes sintéticas;
- scheduler de difusión;
- forward y backward de la U-Net;
- forward y backward de la rama condicional;
- congelamiento de la U-Net base;
- escritura en el volumen de resultados.

El reporte queda persistido en:

```text
/home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird/
└── smoke_test/smoke_test_report.json
```

También es posible probar el modelo explícitamente en CPU:

```bash
python tests/smoke_test.py --device cpu
```

## Entrenamiento

Todos los comandos siguientes se ejecutan dentro del contenedor desde `/app`.

Entrenar la U-Net base:

```bash
python src/Train_Image_Branch.py --config src/default.yaml
```

Entrenar la rama condicional después de generar el checkpoint base:

```bash
python src/Train_Conditional_Partition.py --config src/default.yaml
```

Ejecutar la generación:

```bash
python src/DualSample_ajustable_evalgen_boost.py --config src/default.yaml
```

Ejecutar las tres etapas secuencialmente:

```bash
python src/run.py
```

Los hiperparámetros, rutas internas, batch size y número de épocas se encuentran
en `src/default.yaml`.

## Resultados

Los resultados se escriben bajo `/app/Rislab_Event_influence_volume`, que está
montado en el directorio de salida del host. Entre ellos se encuentran:

```text
Rislab_Event_influence_volume/
├── DDPM/                  # checkpoint y logs de la U-Net base
├── all/                   # entrenamiento condicional general
├── 0/ ... 9/              # entrenamientos condicionales por clase
├── inferencia.log
└── smoke_test/
```

## Licencia

Este repositorio conserva la licencia GNU General Public License v3 del proyecto
`Ebird_MNIST`. Consulta `LICENSE` para conocer sus términos.
