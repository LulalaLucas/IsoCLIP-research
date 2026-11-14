#!/usr/bin/env bash
set -e
cd /data/260010047/lky/IsoCLIP

# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/mixed_projection.json > runs/iso_b32/logs/mixed_projection.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/mixed_center_b8.json > runs/iso_b32/logs/mixed_center_b8.log 2>&1
