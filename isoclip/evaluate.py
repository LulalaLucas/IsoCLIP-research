"""One configuration, all declared datasets; existing results are never overwritten."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .adaptation import fit_transform, DescriptorTransform
from .data import PROJECT, load_prepared, sha256
from .metrics import retrieval
from .projection import energy_operator


def evaluate(cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = PROJECT/cfg['output']
    out.mkdir(parents=True, exist_ok=False)
    prepared = PROJECT/cfg['prepared']
    spectrum = torch.load(prepared/'spectrum.pt', map_location='cpu')
    p, rows = cfg['parameters'], []
    with torch.no_grad():
        for dataset in cfg['datasets']:
            started = time.monotonic()
            data = load_prepared(prepared, dataset)
            assert data['record']['checkpoint_sha256'] == cfg['checkpoint_sha256']
            if p['input_variant'] == 'mixed':
                variant = torch.load(prepared/'mixed'/f'{dataset}.pt', map_location='cpu')
                data.update({k: variant[k] for k in ['query', 'gallery', 'calibration']})
            elif p['input_variant'] == 'gate3':
                gate = torch.load(PROJECT/cfg['gate_checkpoint'], map_location='cpu')['output_operator']
                gm = gate.float().cuda()
                data['query'] = (F.normalize(data['query'].cuda(), dim=1)@gm).cpu()
                data['gallery'] = (F.normalize(data['gallery'].cuda(), dim=1)@gm).cpu()
                data['calibration'] = F.normalize(data['calibration']@gate.double(), dim=1)
            else:
                assert p['input_variant'] == 'iso'
            if p['method'] == 'gate':
                gate = torch.load(PROJECT/cfg['gate_checkpoint'], map_location='cpu')
                state = DescriptorTransform(gate['output_operator'].double(), torch.zeros(512, dtype=torch.float64))
                info = {'method': 'gate', 'output_centered': False, 'output_dimensions': 512,
                    'fit_uses_labels': True, 'gate_source_datasets': gate['evidence']['sources']}
            elif p['method'] == 'energy':
                coordinates = torch.load(prepared/'energy'/f'{dataset}.pt', map_location='cpu')
                basis = torch.load(prepared/'energy_basis.pt', map_location='cpu')
                matrix, indices = energy_operator(coordinates, spectrum, basis, p['rank'])
                state = DescriptorTransform(matrix, torch.zeros(512, dtype=torch.float64))
                info = {'method': 'energy', 'output_centered': False, 'output_dimensions': 512,
                    'effective_rank': p['rank'], 'fit_uses_labels': False, 'selected_iso_axes': indices.tolist()}
            else:
                state, info = fit_transform(data['calibration'], spectrum['output_basis'], p)
            torch.save({**state.state_dict(), 'parameters': p, 'comparison_baseline': 'pure_iso_b32',
                'input_variant': p['input_variant']}, out/f'{dataset}_transform.pt')
            if cfg['identifier'] == 'iso':
                q, g = data['query'].cuda(), data['gallery'].cuda()
            else:
                # Keep the historical arithmetic and normalization order. The
                # public DescriptorTransform also applies the final normalization.
                m, c = state.matrix.float().cuda(), state.center.float().cuda()
                q = (F.normalize(data['query'].cuda(), dim=1)-c)@m
                g = (F.normalize(data['gallery'].cuda(), dim=1)-c)@m
            ap, top, checks = retrieval(q, g, data['query_labels'], data['gallery_labels'], cfg['batch_size'])
            np.savez_compressed(out/f'{dataset}_queries.npz', ap=ap, top10=top,
                labels=data['query_labels'].numpy(), names=np.asarray(data['query_names']))
            row = {'identifier': cfg['identifier'], **data['record'], **info, 'input_variant': p['input_variant'],
                'mAP': float(ap.mean()*100), 'seconds': time.monotonic()-started, 'sklearn_checks': checks}
            rows.append(row)
            (out/'results.partial.json').write_text(json.dumps({'status': 'partial', 'config': cfg, 'rows': rows}, indent=2)+'\n')
            print(f'{cfg["identifier"]} {dataset}: {row["mAP"]:.8f}', flush=True)
            del data, q, g
            torch.cuda.empty_cache()
    report = {'status': 'complete', 'config': cfg, 'rows': rows,
        'source_sha256': {p.name: sha256(p) for p in (PROJECT/'isoclip').glob('*.py')},
        'completed_at': datetime.now(timezone.utc).isoformat()}
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    cfg = json.loads(parser.parse_args().config.read_text())
    evaluate(cfg)


if __name__ == '__main__':
    main()
