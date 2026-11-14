"""Read the benchmark's native annotations without historical experiment files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import names_digest


# Paths follow the dataset layout used by the author's loaders and CoOp.
LAYOUT = {
    'oxford_pets': ('oxford_pets', 'images', 'split_zhou_OxfordPets.json'),
    'stanford_cars': ('stanford_cars', '.', 'split_zhou_StanfordCars.json'),
    'caltech101': ('caltech-101', '101_ObjectCategories', 'split_zhou_Caltech101.json'),
    'eurosat': ('eurosat', '2750', 'split_zhou_EuroSAT.json'),
    'oxford_flowers': ('oxford_flowers', 'jpg', 'split_zhou_OxfordFlowers.json'),
    'dtd': ('dtd', 'images', 'split_zhou_DescribableTextures.json'),
    'ucf101': ('ucf101', 'UCF-101-midframes', 'split_zhou_UCF101.json'),
    'food101': ('food-101', 'images', 'split_zhou_Food101.json'),
    'sun397': ('sun397', 'SUN397', 'split_zhou_SUN397.json'),
}


def read_split(dataroot, dataset, split):
    """Return image paths, cache names and labels in annotation order."""
    root = Path(dataroot)
    if dataset == 'cub2011':
        root = root/'CUB_200_2011'
        labels = dict(line.split() for line in (root/'image_class_labels.txt').read_text().splitlines())
        flags = dict(line.split() for line in (root/'train_test_split.txt').read_text().splitlines())
        rows = [line.split(' ', 1) for line in (root/'images.txt').read_text().splitlines()]
        assert split in ['all', 'train', 'test']
        rows = [row for row in rows if split == 'all' or flags[row[0]] == str(int(split == 'train'))]
        paths = [root/'images'/row[1] for row in rows]
        names = [path.stem for path in paths]
        targets = [int(labels[row[0]]) for row in rows]  # Historical CUB labels are 1-based.
    elif dataset == 'fgvc_aircraft':
        root = root/'fgvc_aircraft'
        classes = (root/'variants.txt').read_text().splitlines()
        rows = [line.split(' ', 1) for line in (root/f'images_variant_{split}.txt').read_text().splitlines()]
        paths = [root/'images'/f'{row[0]}.jpg' for row in rows]
        names = [f'images__{path.name}' for path in paths]
        targets = [classes.index(row[1]) for row in rows]
    else:
        folder, images, annotation = LAYOUT[dataset]
        rows = json.loads((root/folder/annotation).read_text())[split]
        paths = [root/folder/images/row[0] for row in rows]
        names = [f'{path.parent.name}__{path.name}' for path in paths]
        targets = [int(row[1]) for row in rows]
    assert len(names) == len(set(names)), f'{dataset}/{split}: duplicate cache names'
    return {'paths': paths, 'names': names, 'labels': torch.tensor(targets, dtype=torch.long)}


def cub_gallery_order(labels):
    """Keep the PPT's train-only 80/20 concatenation order, retaining every image."""
    labels = labels.numpy()
    rng = np.random.default_rng(42)
    held_out = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        held_out.extend(rng.permutation(members)[:max(1, round(len(members)*.2))])
    held_out = np.array(sorted(held_out))
    retained = np.setdiff1d(np.arange(len(labels)), held_out)
    return np.concatenate([retained, held_out]).tolist()


def identity_fingerprint(names, labels):
    return {'images': len(names), 'names_sha256': names_digest(names),
        'labels_sha256': hashlib.sha256(labels.numpy().astype('<i8').tobytes()).hexdigest()}


def read_identity(dataroot, dataset, protocol):
    gallery, query = read_split(dataroot, dataset, 'train'), read_split(dataroot, dataset, 'test')
    if dataset == 'cub2011':
        order = cub_gallery_order(gallery['labels'])
        gallery['names'] = [gallery['names'][i] for i in order]
        gallery['labels'] = gallery['labels'][order]
    for key, split in [('gallery', gallery), ('query', query)]:
        actual = identity_fingerprint(split['names'], split['labels'])
        assert actual == protocol[dataset][key], f'{dataset}/{key}: annotations differ from the frozen PPT protocol: {actual}'
    assert set(gallery['names']).isdisjoint(query['names'])
    calibration = sorted(gallery['names'], key=lambda n: hashlib.sha256(n.encode()).digest())[:4096]
    assert names_digest(calibration) == protocol[dataset]['calibration_names_sha256']
    return {'query_names': query['names'], 'query_labels': query['labels'],
        'gallery_names': gallery['names'], 'gallery_labels': gallery['labels'],
        'calibration_names': calibration}
