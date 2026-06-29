# Smoke test sin dataset

Este test comprueba el entorno Python, CUDA, los cargadores de datos, el
scheduler y las dos ramas del modelo usando imágenes generadas temporalmente.
No necesita MNIST ni N-MNIST.

Dentro del contenedor:

```bash
python tests/smoke_test.py
```

Para exigir que se use la GPU y fallar si CUDA no está disponible:

```bash
python tests/smoke_test.py --device cuda
```

El resultado también se escribe en:

```text
/app/Rislab_Event_influence_volume/smoke_test/smoke_test_report.json
```

Con `run_docker.sh`, ese archivo quedará persistido en el directorio de salida
del host bajo `smoke_test/smoke_test_report.json`.
