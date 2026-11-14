"""Train the PPT's source-only three-parameter gate; backbone features are frozen."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .data import PROJECT, load_prepared
from .spectral import SpectralGate, class_sampler, sample_batch, nca_loss


def train(cfg):
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = PROJECT/cfg['output']
    out.mkdir(parents=True, exist_ok=False)
    prepared = PROJECT/cfg['prepared']
    spectrum = torch.load(prepared/'spectrum.pt', map_location='cpu')
    right = spectrum['output_basis'].float().cuda()
    output = right.T
    batches, counts = {}, {}
    for ds in cfg['sources']:
        data = load_prepared(prepared, ds)
        coordinates = F.normalize(data['gallery'].cuda(), dim=1)@right
        batches[ds] = class_sampler(coordinates, data['gallery_labels'], 'cuda')
        counts[ds] = len(data['gallery'])
    model = SpectralGate(spectrum['singular_values'], cfg['degree'], 'cuda')
    assert model.theta.numel() == 3
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    torch.cuda.synchronize()
    started = time.monotonic()
    history = []
    for step in range(1, cfg['steps']+1):
        losses = []
        for ds in cfg['sources']:
            coordinates, labels = sample_batch(batches[ds])
            losses.append(nca_loss(model(coordinates, output), labels, cfg['temperature']))
        primary = torch.stack(losses).mean()
        penalty = model.log_weights().square().mean()
        loss = primary+cfg['regularization']*penalty
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(model.theta.grad.norm())
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {'step': step, 'loss': float(loss), 'nca': float(primary), 'penalty': float(penalty),
                'gradient_norm': gradient, 'theta': model.theta.detach().cpu().tolist()}
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    gates = model.weights().detach().double().cpu()
    right = spectrum['output_basis']
    operator = (right*gates[None, :])@right.T
    assert gates.min() >= .5-1e-6 and gates.max() <= 2+1e-6
    assert abs(float(gates.log().mean())) < 1e-6
    assert {int(s['step']) for s in optimizer.state.values()} == {cfg['steps']}
    evidence = {'backbone': 'OpenAI CLIP ViT-B/32', 'base': 'pure IsoCLIP rank312',
        'sources': cfg['sources'], 'native_train_images': counts, 'parameters': 3,
        'optimizer_updates': cfg['steps'], 'fit_seconds': time.monotonic()-started,
        'target_images_used_for_fit': 0, 'target_labels_used_for_fit': 0, 'backbone_updates': 0,
        'checkpoint_selection': 'Fixed2000steps; reproduces the final PPT training configuration.',
        'gpu': torch.cuda.get_device_name(0), 'seed': cfg['seed']}
    torch.save({'theta': model.theta.detach().cpu(), 'gates': gates, 'output_operator': operator,
        'full_projection': spectrum['projection']@operator, 'optimizer': optimizer.state_dict(),
        'config': cfg, 'history': history, 'evidence': evidence}, out/'gate.pt')
    (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    (out/'completed.json').write_text(json.dumps({'status': 'complete', 'evidence': evidence,
        'completed_at': datetime.now(timezone.utc).isoformat()}, indent=2)+'\n')
    return model.theta.detach().cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    train(json.loads(parser.parse_args().config.read_text()))
