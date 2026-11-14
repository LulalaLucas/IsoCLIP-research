"""Cached descriptors and exact train/test identities for the PPT benchmark."""
import hashlib
from pathlib import Path
import pickle

import torch

PROJECT = Path(__file__).resolve().parents[1]
STUDY = PROJECT/'reproduction/research/iso_baseline_study'
DATASETS = ['cub2011', 'oxford_pets', 'stanford_cars', 'fgvc_aircraft', 'caltech101',
    'eurosat', 'oxford_flowers', 'dtd', 'ucf101', 'food101', 'sun397']
SOURCES = ['cub2011', 'oxford_pets', 'stanford_cars']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def names_digest(names):
    return hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()


def load_prepared(prepared, dataset):
    data = torch.load(Path(prepared)/f'{dataset}.pt', map_location='cpu')
    assert set(data['query_names']).isdisjoint(data['gallery_names'])
    assert set(data['calibration_names']).issubset(data['gallery_names'])
    expected = sorted(data['gallery_names'], key=lambda n: hashlib.sha256(n.encode()).digest())[:4096]
    assert data['calibration_names'] == expected
    for key in ['query', 'gallery', 'calibration']:
        assert names_digest(data[f'{key}_names']) == data['record'][f'{key}_names_sha256']
    assert not data['record']['test_used_for_fit']
    return data


def raw_features(dataset, split, names):
    """Read existing real backbone features in the explicit frozen name order."""
    storage = 'all' if dataset == 'cub2011' else split
    directory = PROJECT/'data/image_noproj_features'/dataset/'ViT-B32'/storage
    raw = torch.load(directory/'image_noproj_features.pt', map_location='cpu').float()
    with (directory/'image_noproj_names.pkl').open('rb') as stream:
        cached_names = pickle.load(stream)
    assert len(cached_names) == len(set(cached_names)) == len(raw)
    lookup = {n: i for i, n in enumerate(cached_names)}
    return raw[[lookup[n] for n in names]]
