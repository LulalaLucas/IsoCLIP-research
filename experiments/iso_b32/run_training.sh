#!/usr/bin/env bash
set -e
cd /data/260010047/lky/IsoCLIP

# CUDA_VISIBLE_DEVICES=0 /data/260010047/lky/envs/isoclip/bin/python -u -m isoclip.train --config experiments/iso_b32/configs/gate_training.json > runs/iso_b32/logs/gate_training.log 2>&1
