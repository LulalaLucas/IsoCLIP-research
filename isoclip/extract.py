"""Encode real images into the author's B/32 pre-projection feature-cache format."""
import argparse
import json
from pathlib import Path
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .data import PROJECT, sha256
from .dataset_splits import read_identity, read_split


class Images(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image)


@torch.no_grad()
def extract(cfg):
    import clip

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # Same image-encoding setting as src/utils.py.
    dataroot, cache = PROJECT/cfg['dataroot'], PROJECT/cfg['feature_cache']
    protocol = json.loads((PROJECT/cfg['protocol']).read_text())['datasets']
    jobs = []
    for dataset in cfg['datasets']:
        read_identity(dataroot, dataset, protocol)
        splits = ['all'] if dataset == 'cub2011' else ['train', 'test']
        for split in splits:
            samples = read_split(dataroot, dataset, split)
            directory = cache/dataset/'ViT-B32'/split
            feature_file = directory/'image_noproj_features.pt'
            if feature_file.exists():
                with (directory/'image_noproj_names.pkl').open('rb') as stream:
                    names = pickle.load(stream)
                assert names == samples['names'], f'{dataset}/{split}: cached names differ from annotations'
                features = torch.load(feature_file, map_location='cpu')
                assert features.shape == (len(names), 768)
                assert torch.isfinite(features).all()
                print(f'CACHED {dataset}/{split}: {len(names)} images', flush=True)
                del features
            else:
                jobs.append((dataset, split, directory, samples))
    checkpoint = PROJECT/cfg['checkpoint']
    assert checkpoint.name == 'ViT-B-32.pt'
    model, preprocess = clip.load('ViT-B/32', device='cuda', jit=False, download_root=str(checkpoint.parent))
    assert sha256(checkpoint) == cfg['checkpoint_sha256']
    if not jobs:
        return
    model = model.float().eval().requires_grad_(False)
    model.visual.proj = None  # Return ln_post(CLS), exactly as the author's no-projection encoder.
    for dataset, split, directory, samples in jobs:
        loader = DataLoader(Images(samples['paths'], preprocess), batch_size=cfg['batch_size'],
            num_workers=cfg['workers'], shuffle=False, pin_memory=True)
        encoded = [model.encode_image(batch.cuda()).cpu() for batch in tqdm(loader, desc=f'{dataset}/{split}')]
        features = torch.cat(encoded)
        assert features.shape == (len(samples['names']), 768)
        assert torch.isfinite(features).all()
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'image_noproj_names.pkl').open('wb') as stream:
            pickle.dump(samples['names'], stream)
        torch.save(features, directory/'image_noproj_features.pt')
        print(f'SAVED {dataset}/{split}: {len(features)} images', flush=True)
        del features, encoded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    extract(json.loads(parser.parse_args().config.read_text()))
