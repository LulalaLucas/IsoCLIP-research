#!/usr/bin/env bash
set -e
cd /data/260010047/lky/IsoCLIP

# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b8_seed20261106.json > runs/iso_b32/logs/b8_seed20261106.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b8_seed20261107.json > runs/iso_b32/logs/b8_seed20261107.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8_seed20261106.json > runs/iso_b32/logs/center_b8_seed20261106.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8_seed20261107.json > runs/iso_b32/logs/center_b8_seed20261107.log 2>&1
