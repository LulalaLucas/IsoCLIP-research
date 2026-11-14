#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u -m isoclip.extract --config experiments/iso_b32/configs/prepare.json
python -u -m isoclip.prepare --config experiments/iso_b32/configs/prepare.json
python -u -m isoclip.train --config experiments/iso_b32/configs/gate_training.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/iso.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/mean_axis.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/pca90.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/whiten.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/energy256.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/local_qr.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/soft25.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b8.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/gate3.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b2.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/centered_neighbors.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/hubness.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/shuffled_graph.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_pca90.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_whiten.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_local_qr.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/gate_center_b8.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b8_seed20261106.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/b8_seed20261107.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8_seed20261106.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/center_b8_seed20261107.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/mixed_projection.json
python -u -m isoclip.evaluate --config experiments/iso_b32/configs/mixed_center_b8.json
