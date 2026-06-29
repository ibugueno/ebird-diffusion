#!/bin/bash

docker rm -fv ignacio_event_ebird

docker run -it \
  --gpus all \
  --name ignacio_event_ebird \
  --shm-size=224g \
  -v /home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird:/app/Rislab_Event_influence_volume \
  -v /home/ignacio.bugueno/cachefs/datasets/processed_data/reconstruction:/app/Rislab_Event_influence_volume/dataset:ro \
  ignacio_event_ebird
