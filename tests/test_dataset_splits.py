"""Protect native split boundaries, labels, order and frozen protocol checks."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from isoclip.data import names_digest
from isoclip.dataset_splits import cub_gallery_order, identity_fingerprint, read_identity, read_split


class SplitTests(unittest.TestCase):
    def test_cub_respects_official_flags_and_historical_label_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'CUB_200_2011'
            root.mkdir()
            (root/'images.txt').write_text('1 001.Bird/a.jpg\n2 001.Bird/b.jpg\n3 002.Bird/c.jpg\n')
            (root/'image_class_labels.txt').write_text('1 1\n2 1\n3 2\n')
            (root/'train_test_split.txt').write_text('1 1\n2 0\n3 1\n')
            train = read_split(directory, 'cub2011', 'train')
            test = read_split(directory, 'cub2011', 'test')
            self.assertEqual(train['names'], ['a', 'c'])
            self.assertEqual(train['labels'].tolist(), [1, 2])
            self.assertEqual(test['names'], ['b'])
            self.assertEqual(len(read_split(directory, 'cub2011', 'all')['names']), 3)

    def test_cub_reordering_keeps_every_native_training_sample_once(self):
        order = cub_gallery_order(torch.tensor([1]*5+[2]*5))
        self.assertEqual(order, [0, 1, 2, 3, 5, 6, 7, 9, 4, 8])
        self.assertEqual(sorted(order), list(range(10)))

    def test_protocol_rejects_changed_labels_even_when_names_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'oxford_pets'
            root.mkdir()
            rows = {'train': [['cat.jpg', 0, 'cat'], ['dog.jpg', 1, 'dog']],
                'test': [['kitten.jpg', 0, 'cat']]}
            path = root/'split_zhou_OxfordPets.json'
            path.write_text(json.dumps(rows))
            protocol = {'oxford_pets': {
                'gallery': identity_fingerprint(['images__cat.jpg', 'images__dog.jpg'], torch.tensor([0, 1])),
                'query': identity_fingerprint(['images__kitten.jpg'], torch.tensor([0])),
                'calibration_names_sha256': names_digest(['images__dog.jpg', 'images__cat.jpg'])}}
            identity = read_identity(directory, 'oxford_pets', protocol)
            self.assertEqual(identity['gallery_names'], ['images__cat.jpg', 'images__dog.jpg'])
            rows['train'][0][1] = 1
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(AssertionError, 'annotations differ'):
                read_identity(directory, 'oxford_pets', protocol)


if __name__ == '__main__':
    unittest.main()
