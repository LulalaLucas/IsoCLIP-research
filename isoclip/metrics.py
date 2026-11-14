"""Full-ranking class retrieval AP; query and gallery must be disjoint."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


@torch.no_grad()
def retrieval(query, gallery, query_labels, gallery_labels, batch_size):
    q, g = F.normalize(query, dim=1), F.normalize(gallery, dim=1)
    qlabels, glabels = query_labels.to(q.device), gallery_labels.to(q.device)
    positions = torch.arange(1, len(g)+1, device=q.device, dtype=torch.float64)
    check_ids = set(np.linspace(0, len(q)-1, min(16, len(q)), dtype=int).tolist())
    aps, tops, checks = [], [], []
    for start in range(0, len(q), batch_size):
        scores = q[start:start+batch_size]@g.T
        order = scores.topk(len(g), dim=1, sorted=True).indices
        relevant = (glabels[order] == qlabels[start:start+len(order), None]).double()
        count = relevant.sum(1)
        assert (count > 0).all()
        ap = (relevant.cumsum(1)/positions*relevant).sum(1)/count
        for local in range(len(order)):
            if start+local in check_ids:
                expected = average_precision_score(relevant[local].cpu().numpy(), -np.arange(len(g)))
                error = abs(expected-float(ap[local]))
                assert error < 1e-10
                checks.append({'query': start+local, 'sklearn_fixed_order_error': error})
        aps.append(ap.cpu().numpy())
        tops.append(order[:, :10].cpu().numpy())
    return np.concatenate(aps), np.concatenate(tops), checks
