"""Unlabeled fitting takes training descriptors only; output centering is explicit."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DescriptorTransform:
    matrix: torch.Tensor
    center: torch.Tensor

    def __call__(self, projected_features):
        """Input is the projected Iso descriptor, before normalization."""
        matrix = self.matrix.to(projected_features)
        center = self.center.to(projected_features)
        return F.normalize((F.normalize(projected_features, dim=1)-center)@matrix, dim=1)

    def state_dict(self):
        return {'matrix': self.matrix, 'center': self.center}


@torch.no_grad()
def graph_basis(z, graph, device='cuda'):
    """Frozen PPT rule: mutual10, local shrinkage.3, generalized lambda>1, QR."""
    centered = z-z.mean(0)
    total = centered.T@centered/len(z)
    gpu = z.float().to(device)
    if graph == 'centered':
        gpu = F.normalize(gpu-gpu.mean(0), dim=1)
    scores = gpu@gpu.T
    scores.fill_diagonal_(-torch.inf)
    if graph == 'hubness':
        density = scores.topk(10, dim=1).values.mean(1)
        scores = 2*scores-density[:, None]-density[None, :]
    assert graph in ['cosine', 'centered', 'hubness', 'shuffled']
    neighbors = scores.topk(10, dim=1).indices
    adjacency = torch.zeros_like(scores, dtype=torch.bool)
    adjacency.scatter_(1, neighbors, True)
    edge_i, edge_j = torch.where(torch.triu(adjacency & adjacency.T, diagonal=1))
    edge_i, edge_j = edge_i.cpu(), edge_j.cpu()
    if graph == 'shuffled':
        permutation = torch.randperm(len(z), generator=torch.Generator().manual_seed(42))
        edge_i, edge_j = permutation[edge_i], permutation[edge_j]
    assert len(edge_i) > 0
    difference = z[edge_i]-z[edge_j]
    local = difference.T@difference/(2*len(difference))
    dimension = z.shape[1]
    local = .7*local+.3*local.trace()/dimension*torch.eye(dimension, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(local)
    inverse_half = (vectors*values.rsqrt())@vectors.T
    ratios, vectors = torch.linalg.eigh(inverse_half@total@inverse_half)
    rank = int((ratios > 1).sum())
    assert 0 < rank < dimension
    q = torch.linalg.qr(inverse_half@vectors.flip(1)[:, :rank], mode='reduced').Q
    return q, {'mutual_edges': len(edge_i), 'selected_subspace_dimensions': rank}


@torch.no_grad()
def fit_transform(calibration, output_basis, parameters, graph_device='cuda'):
    """calibration must already be normalized training descriptors, CPU float64.

    Computing a covariance mean does not enable output centering. That requires
    parameters['output_center']=True, including for PCA, whitening and B8.
    """
    z = calibration.double()
    mean = z.mean(0)
    dimension = z.shape[1]
    identity = torch.eye(dimension, dtype=torch.float64)
    method = parameters['method']
    info = {'method': method, 'output_centered': parameters['output_center'], 'fit_uses_labels': False}
    if method == 'identity':
        matrix = identity
    elif method == 'mean_axis':
        direction = mean/mean.norm()
        matrix = identity-torch.outer(direction, direction)
    elif method in ['pca', 'whiten']:
        compact = (z-mean)@output_basis
        total = compact.T@compact/len(z)
        values, vectors = torch.linalg.eigh(total)
        values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
        if method == 'pca':
            rank = int(torch.searchsorted(values.cumsum(0)/values.sum(), parameters['energy']))+1
            matrix = output_basis@vectors[:, :rank]
            info['retained_variance'] = float(values[:rank].sum()/values.sum())
        else:
            regularized = (1-parameters['rho'])*values+parameters['rho']*values.mean()
            matrix = (output_basis@vectors)*regularized.rsqrt()[None, :]
            info['covariance_shrinkage'] = parameters['rho']
    elif method in ['graph', 'soft']:
        matrix, details = graph_basis(z, parameters['graph'], graph_device)
        info.update(details)
        if method == 'soft':
            projector = matrix@matrix.T
            matrix = projector+parameters['complement']**.5*(identity-projector)
    elif method == 'consensus':
        generator = torch.Generator().manual_seed(parameters['split_seed'])
        average = torch.zeros_like(identity)
        ranks = []
        for _ in range(parameters['partitions']):
            permutation = torch.randperm(len(z), generator=generator)
            for indices in [permutation[::2], permutation[1::2]]:
                q, _ = graph_basis(z[indices], 'cosine', graph_device)
                average += q@q.T
                ranks.append(q.shape[1])
        average /= 2*parameters['partitions']
        values, vectors = torch.linalg.eigh(average)
        matrix = vectors[:, values > parameters['threshold']]
        assert matrix.shape[1] > 0
        info.update(subspaces=2*parameters['partitions'], half_ranks=ranks,
            support_threshold=parameters['threshold'], split_seed=parameters['split_seed'])
    else:
        raise ValueError(method)
    center = mean if parameters['output_center'] else torch.zeros_like(mean)
    info['output_dimensions'] = matrix.shape[1]
    return DescriptorTransform(matrix, center), info
