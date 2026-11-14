#!/usr/bin/env bash
set -e
cd /data/260010047/lky/IsoCLIP

# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b2.json > runs/iso_b32/logs/b2.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b8.json > runs/iso_b32/logs/b8.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b8_seed20261106.json > runs/iso_b32/logs/b8_seed20261106.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b8_seed20261107.json > runs/iso_b32/logs/b8_seed20261107.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center.json > runs/iso_b32/logs/center.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8.json > runs/iso_b32/logs/center_b8.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8_seed20261106.json > runs/iso_b32/logs/center_b8_seed20261106.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8_seed20261107.json > runs/iso_b32/logs/center_b8_seed20261107.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_local_qr.json > runs/iso_b32/logs/center_local_qr.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_pca90.json > runs/iso_b32/logs/center_pca90.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_whiten.json > runs/iso_b32/logs/center_whiten.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/centered_neighbors.json > runs/iso_b32/logs/centered_neighbors.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/energy256.json > runs/iso_b32/logs/energy256.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/gate3.json > runs/iso_b32/logs/gate3.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/gate_center_b8.json > runs/iso_b32/logs/gate_center_b8.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/hubness.json > runs/iso_b32/logs/hubness.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/local_qr.json > runs/iso_b32/logs/local_qr.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/mean_axis.json > runs/iso_b32/logs/mean_axis.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/mixed_center_b8.json > runs/iso_b32/logs/mixed_center_b8.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/mixed_projection.json > runs/iso_b32/logs/mixed_projection.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/pca90.json > runs/iso_b32/logs/pca90.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/shuffled_graph.json > runs/iso_b32/logs/shuffled_graph.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/soft25.json > runs/iso_b32/logs/soft25.log 2>&1
# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.evaluate --config experiments/iso_b32/configs/whiten.json > runs/iso_b32/logs/whiten.log 2>&1
