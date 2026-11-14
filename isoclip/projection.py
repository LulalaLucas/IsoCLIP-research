"""Projection matrices use row-vector convention: features @ projection."""
import torch


@torch.no_grad()
def image_projection(image, text, top, bottom):
    """Original IsoCLIP; text is CLIP's untransposed text_projection tensor."""
    image, text = image.double(), text.double()
    u, _, _ = torch.linalg.svd(image@text.T, full_matrices=False)
    assert 0 <= top < len(u.T)-bottom
    middle = u[:, top:len(u.T)-bottom]
    return middle@(middle.T@image)


@torch.no_grad()
def mixed_projection(image, text, top, bottom, alpha):
    """Explicit comparison only: replace the original text Gram prior."""
    image, text = image.double(), text.double()
    gram = text.T@text
    gram = alpha*gram+(1-alpha)*gram.trace()/len(gram)*torch.eye(len(gram), dtype=torch.float64)
    values, vectors = torch.linalg.eigh(gram)
    u, _, _ = torch.linalg.svd(image@((vectors*values.sqrt())@vectors.T), full_matrices=False)
    middle = u[:, top:len(u.T)-bottom]
    return middle@(middle.T@image)


@torch.no_grad()
def projection_spectrum(projection, rank):
    left, singular, right_t = torch.linalg.svd(projection.double(), full_matrices=False)
    return {'projection': projection.double(), 'input_basis': left[:, :rank],
        'singular_values': singular[:rank], 'output_basis': right_t[:rank].T.contiguous(),
        'training_coordinates': left[:, :rank]*singular[:rank][None, :]}


@torch.no_grad()
def energy_operator(coordinates, spectrum, energy_basis, rank):
    """Select only among the cross-modal axes already retained by pure IsoCLIP."""
    scores = coordinates.var(0, unbiased=False)/coordinates.square().mean(0).clamp_min(1e-12)
    indices = scores.topk(rank).indices
    basis = energy_basis['middle'][:, indices]
    filtered = basis@(basis.T@energy_basis['image'])
    matrix = energy_basis['iso_pinv']@filtered
    torch.testing.assert_close(spectrum['projection']@matrix, filtered, rtol=1e-9, atol=1e-10)
    return matrix, indices
