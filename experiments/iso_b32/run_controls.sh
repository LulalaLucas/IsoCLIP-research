#!/usr/bin/env bash
set -e
cd /data/260010047/lky/IsoCLIP

# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b2.json > runs/iso_b32/logs/b2.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/centered_neighbors.json > runs/iso_b32/logs/centered_neighbors.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/hubness.json > runs/iso_b32/logs/hubness.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/shuffled_graph.json > runs/iso_b32/logs/shuffled_graph.log 2>&1
