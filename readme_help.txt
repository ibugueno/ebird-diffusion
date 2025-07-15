1. Construir la imagen Docker
docker build -t training_ec .

2. Ejecutar con volumen conectado a la carpeta Rislab_Event_influence_volume del contenedor.
docker run --rm --gpus "device=0" --name container_training_ec -v Rislab_Event_influence_volume:/app/Rislab_Event_influence_volume training_ec conda run -n EVDiff python /app/src/run.py


/// Para Debugear ///

Ejecutar para debugear a con conexion a una carpeta local y Volumen conectado a la carpeta Rislab_Event_influence_volume del contenedo
docker run -it --gpus "device=0" --name container_training_ec -v C:/Users/Pollo/Documents/Docker_fbn/RISLAB_Event_Influence/Train/src:/app/src -v Rislab_Event_influence_volume:/app/Rislab_Event_influence_volume training_ec conda run -n EVDiff python /app/src/run.py

#python src/Train.py --config 'src/default.yaml'
#python src/sample_ddpm.py --config 'src/default.yaml'

#python src/Conditional_Train.py --config 'src/default.yaml'
#python src/DualSample.py --config 'src/default.yaml'
