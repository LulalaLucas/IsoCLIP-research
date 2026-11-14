"""Rebuild pure-Iso descriptors from real cached image features and frozen identities."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import PROJECT, load_prepared, raw_features, sha256
from .projection import image_projection, mixed_projection, projection_spectrum


@torch.no_grad()
def prepare(cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = PROJECT/cfg['identity_reference']
    out = PROJECT/cfg['output']
    out.mkdir(parents=True, exist_ok=False)
    (out/'mixed').mkdir()
    (out/'energy').mkdir()
    checkpoint = PROJECT/cfg['checkpoint']
    assert sha256(checkpoint) == cfg['checkpoint_sha256']
    weights = torch.jit.load(str(checkpoint), map_location='cpu').state_dict()
    image, text = weights['visual.proj'].double(), weights['text_projection'].double()
    projection = image_projection(image, text, 150, 50)
    spectrum = projection_spectrum(projection, 312)
    torch.save(spectrum, out/'spectrum.pt')
    u, _, _ = torch.linalg.svd(image@text.T, full_matrices=False)
    middle = u[:, 150:462]
    pinv = (spectrum['output_basis']/spectrum['singular_values'][None, :])@spectrum['input_basis'].T
    torch.save({'middle': middle, 'image': image, 'iso_pinv': pinv}, out/'energy_basis.pt')
    mixed = mixed_projection(image, text, 150, 50, .5)
    p, mp = projection.float().cuda(), mixed.float().cuda()
    records = []
    for ds in cfg['datasets']:
        identity = load_prepared(source, ds)
        gallery = raw_features(ds, 'train', identity['gallery_names'])
        query = raw_features(ds, 'test', identity['query_names'])
        lookup = {n: i for i, n in enumerate(identity['gallery_names'])}
        ids = [lookup[n] for n in identity['calibration_names']]
        raw = gallery[ids].double()
        identity['query'] = (query.cuda()@p).cpu()
        identity['gallery'] = (gallery.cuda()@p).cpu()
        identity['calibration'] = F.normalize(raw@projection, dim=1)
        torch.save(identity, out/f'{ds}.pt')
        torch.save({'query': (query.cuda()@mp).cpu(), 'gallery': (gallery.cuda()@mp).cpu(),
            'calibration': F.normalize(raw@mixed, dim=1)}, out/'mixed'/f'{ds}.pt')
        torch.save(raw@middle, out/'energy'/f'{ds}.pt')
        records.append(identity['record'])
        print(f'PREPARED {ds}: query={len(query)},gallery={len(gallery)},calibration={len(ids)}', flush=True)
        del gallery, query, identity
        torch.cuda.empty_cache()
    (out/'manifest.json').write_text(json.dumps({'status': 'complete', 'config': cfg, 'datasets': records}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    prepare(json.loads(parser.parse_args().config.read_text()))
