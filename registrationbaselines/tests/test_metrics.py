import unittest
from pathlib import Path
import sys

import torch


sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

import registrationbaselines.core.metrics as metrics


class TestDICE(unittest.TestCase):

    def test_dice_score_identical(self):
        """Test case where both volumes are identical, so Dice score should be 1."""
        image1 = torch.tensor([
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        ])

        image2 = image1.clone()  # Identical to image1

        expected_scores = [1.0]  # One score - we ignore 0 class
        actual_scores = metrics.dice_score(image1, image2)
        self.assertEqual(actual_scores, expected_scores)

    def test_dice_score_ignore_zero_class(self):
        """Test case where both volumes are identical, so Dice score should be 1."""
        image1 = torch.tensor([
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        ])

        image2 = image1.clone()  # Identical to image1

        len_expected_scores = 1  # One score - we ignore 0 class
        actual_scores = metrics.dice_score(image1, image2)
        self.assertEqual(len(actual_scores), len_expected_scores)

    def test_dice_score_no_overlap(self):
        """Test case where volumes have no overlap, so Dice score should be 0."""
        image1 = torch.tensor([
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        ])

        image2 = torch.tensor([
            [[1, 1, 1], [0, 0, 0], [1, 1, 1]],
            [[1, 1, 1], [0, 0, 0], [1, 1, 1]],
            [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
        ])

        expected_scores = [0.0]  # One score for each class (0 and 1)
        actual_scores = metrics.dice_score(image1, image2)
        self.assertEqual(actual_scores, expected_scores)

        image1 = torch.tensor([
            [[2, 2, 2], [1, 1, 1], [0, 0, 0]],
            [[2, 2, 2], [1, 1, 1], [0, 0, 0]],
            [[2, 2, 2], [1, 1, 1], [0, 0, 0]]
        ])

        image2 = torch.tensor([
            [[1, 1, 1], [0, 0, 0], [2, 2, 2]],
            [[1, 1, 1], [0, 0, 0], [2, 2, 2]],
            [[1, 1, 1], [0, 0, 0], [2, 2, 2]]
        ])

        expected_scores = [0.0, 0.0]  # One score for each class (0 and 1)
        actual_scores = metrics.dice_score(image1, image2)
        self.assertEqual(actual_scores, expected_scores)

    def test_dice_score_partial_overlap(self):
        """Test case where volumes have partial overlap."""
        image1 = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 0, 1], [1, 0, 0]]
        ])

        image2 = torch.tensor([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 1]]
        ])

        # we have 2 intersections and in both the top and bottom we have 4 values, so 2 * 2 / (4+4) = 0.5

        expected_scores = [0.5]  # Dice score should be 0.5
        actual_scores = metrics.dice_score(image1, image2)
        self.assertEqual(actual_scores, expected_scores)
