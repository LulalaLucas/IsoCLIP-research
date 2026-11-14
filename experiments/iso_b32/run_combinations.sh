#!/usr/bin/env bash
set -e
cd /data/260010047/lky/IsoCLIP

# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_pca90.json > runs/iso_b32/logs/center_pca90.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_whiten.json > runs/iso_b32/logs/center_whiten.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_local_qr.json > runs/iso_b32/logs/center_local_qr.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8.json > runs/iso_b32/logs/center_b8.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/gate_center_b8.json > runs/iso_b32/logs/gate_center_b8.log 2>&1
