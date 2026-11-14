"""Mathematical invariants protecting pure-Iso and explicit strategy composition."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from isoclip.adaptation import fit_transform
from isoclip.metrics import retrieval
from isoclip.projection import image_projection
from isoclip.spectral import SpectralGate


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_original_projection_uses_transposed_text_head(self):
        image, text = torch.randn(19, 12).double(), torch.randn(12, 12).double()
        actual = image_projection(image, text, 3, 2)
        u, _, _ = torch.linalg.svd(image@text.T, full_matrices=False)
        expected = u[:, 3:10]@(u[:, 3:10].T@image)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(int(torch.linalg.matrix_rank(actual)), 7)

    def test_pca_centering_switch_leaves_fitted_directions_unchanged(self):
        z = F.normalize(torch.randn(80, 12).double()+torch.arange(12).double()/5, dim=1)
        basis = torch.eye(12, dtype=torch.float64)
        p = {'method': 'pca', 'energy': .9, 'output_center': False}
        single, _ = fit_transform(z, basis, p)
        combined, _ = fit_transform(z, basis, {**p, 'output_center': True})
        torch.testing.assert_close(single.matrix, combined.matrix, atol=0, rtol=0)
        self.assertEqual(float(single.center.norm()), 0.)
        torch.testing.assert_close(combined.center, z.mean(0))
        self.assertFalse(torch.allclose(single(z), combined(z)))

    def test_whitening_operates_only_in_supplied_retained_space(self):
        basis = torch.linalg.qr(torch.randn(12, 7).double(), mode='reduced').Q
        z = F.normalize(torch.randn(80, 7).double()@basis.T, dim=1)
        state, _ = fit_transform(z, basis, {'method': 'whiten', 'rho': .1, 'output_center': False})
        self.assertEqual(tuple(state.matrix.shape), (12, 7))
        torch.testing.assert_close(basis@(basis.T@state.matrix), state.matrix, atol=1e-12, rtol=1e-10)

    def test_mean_subtraction_differs_from_removing_mean_axis(self):
        z = F.normalize(torch.randn(80, 12).double()+2, dim=1)
        a, _ = fit_transform(z, torch.eye(12).double(), {'method': 'identity', 'output_center': True})
        b, _ = fit_transform(z, torch.eye(12).double(), {'method': 'mean_axis', 'output_center': False})
        torch.testing.assert_close(b.matrix@b.matrix, b.matrix, atol=1e-12, rtol=1e-10)
        self.assertFalse(torch.allclose(a(z), b(z)))

    def test_spectral_zero_and_bounded_positive_weights(self):
        gate = SpectralGate(torch.linspace(3, .1, 37), 3, 'cpu')
        torch.testing.assert_close(gate.weights(), torch.ones(37), atol=0, rtol=0)
        with torch.no_grad():
            gate.theta.copy_(torch.tensor([20., -10., 30.]))
        self.assertTrue(.5 <= float(gate.weights().min()) <= float(gate.weights().max()) <= 2)
        self.assertLess(abs(float(gate.log_weights().mean())), 1e-6)

    def test_full_ranking_ap_matches_hand_calculation(self):
        q = torch.tensor([[1., 0.]])
        gallery = torch.tensor([[1., 0.], [.8, .6], [0., 1.]])
        ap, top, checks = retrieval(q, gallery, torch.tensor([0]), torch.tensor([0, 1, 0]), 2)
        self.assertAlmostEqual(ap[0], (1+2/3)/2)
        np.testing.assert_array_equal(top[0], [0, 1, 2])
        self.assertEqual(len(checks), 1)


if __name__ == '__main__':
    unittest.main()
