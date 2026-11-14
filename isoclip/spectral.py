"""Source-supervised three-parameter curve and frozen PK/NCA training rules."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class SpectralGate(nn.Module):
    def __init__(self, singular_values, degree, device):
        super().__init__()
        log_s = singular_values.float().to(device).log()
        x = 1-2*(log_s-log_s.min())/(log_s.max()-log_s.min())
        previous, current = torch.ones_like(x), x
        columns = [current]
        for k in range(1, degree):
            following = ((2*k+1)*x*current-k*previous)/(k+1)
            columns.append(following)
            previous, current = current, following
        basis = torch.stack(columns, dim=1)
        basis = basis-basis.mean(0)
        basis = basis/basis.square().mean(0).sqrt()
        self.register_buffer('basis', basis)
        self.theta = nn.Parameter(torch.zeros(degree, device=device))

    def log_weights(self):
        values = .5*math.log(2)*torch.tanh(self.basis@self.theta)
        return values-values.mean()

    def weights(self):
        return self.log_weights().exp()

    def forward(self, coordinates, output_matrix):
        return F.normalize((coordinates*self.weights())@output_matrix, dim=1)


def nca_loss(features, labels, temperature):
    logits = features@features.T/temperature
    diagonal = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = (labels[:, None] == labels[None, :]) & ~diagonal
    assert positives.any(1).all()
    denominator = logits.masked_fill(diagonal, -torch.inf).logsumexp(1)
    numerator = logits.masked_fill(~positives, -torch.inf).logsumexp(1)
    return (denominator-numerator).mean()


def class_sampler(coordinates, labels, device):
    members = [torch.where(labels == label)[0] for label in labels.unique()]
    maximum = max(len(m) for m in members)
    assert min(len(m) for m in members) >= 8 and len(members) >= 16
    indices = torch.zeros(len(members), maximum, dtype=torch.long, device=device)
    lengths = torch.tensor([len(m) for m in members], device=device)
    for i, rows in enumerate(members):
        indices[i, :len(rows)] = rows.to(device)
    return {'coordinates': coordinates.to(device), 'members': indices, 'lengths': lengths}


def sample_batch(state):
    device = state['members'].device
    classes = torch.randperm(len(state['members']), device=device)[:16]
    random = torch.rand(16, state['members'].shape[1], device=device)
    invalid = torch.arange(random.shape[1], device=device)[None, :] >= state['lengths'][classes, None]
    positions = random.masked_fill(invalid, -1).topk(8, dim=1).indices
    indices = state['members'][classes].gather(1, positions).flatten()
    assert len(indices.unique()) == 128
    return state['coordinates'][indices], torch.arange(16, device=device).repeat_interleave(8)
