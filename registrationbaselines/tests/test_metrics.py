import math
import unittest
from pathlib import Path
import sys

from typing import List

import torch
import numpy as np


sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

import registrationbaselines.metrics.metrics as metrics


def compare_lists_with_nan(list1: List[float], list2: List[float]) -> bool:
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if np.isnan(a) and np.isnan(b):
            continue
        if abs(a - b) > 1e-5:
            return False
    return True


class TestDICE(unittest.TestCase):

    def test_dice_score_identical(self):
        """Test case where both volumes are identical, so Dice score should be 1."""
        image1 = torch.tensor([
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        ]).to(torch.uint8)

        image2 = image1.clone()  # Identical to image1

        expected_scores = [1.0]  # One score - we ignore 0 class
        ours, ours_mean = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        # monai = metrics.dice_score_monai(image1.unsqueeze(
        #     0).unsqueeze(0), image2.unsqueeze(0).unsqueeze(0))

        self.assertEqual([ours['1']], expected_scores)
        self.assertEqual(l2r, expected_scores)
        # self.assertEqual(monai, expected_scores)

    def test_dice_score_no_overlap(self):
        """Test case where volumes have no overlap, so Dice score should be 0."""
        image1 = torch.tensor([
            [[2, 2, 2], [1, 1, 1], [2, 2, 2]],
            [[2, 2, 2], [1, 1, 1], [2, 2, 2]],
            [[2, 2, 2], [1, 1, 1], [2, 2, 2]]
        ]).to(torch.uint8)

        image2 = torch.tensor([
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]],
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]],
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]]
        ]).to(torch.uint8)

        expected_scores = [0.0, 0.0]  # One score for each class (1 and 2)
        ours, ours_mean = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        # monai = metrics.dice_score_monai(image1, image2)

        self.assertEqual([ours['1'], ours['2']], expected_scores)
        self.assertEqual(l2r, expected_scores)
        # self.assertEqual(monai, expected_scores)

        image1 = torch.tensor([
            [[2, 2, 2], [1, 1, 1], [0, 0, 0]],
            [[2, 2, 2], [1, 1, 1], [0, 0, 0]],
            [[2, 2, 2], [1, 1, 1], [0, 0, 0]]
        ]).to(torch.uint8)

        image2 = torch.tensor([
            [[1, 1, 1], [0, 0, 0], [2, 2, 2]],
            [[1, 1, 1], [0, 0, 0], [2, 2, 2]],
            [[1, 1, 1], [0, 0, 0], [2, 2, 2]]
        ]).to(torch.uint8)

        expected_scores = [0.0, 0.0]  # One score for each class (0 and 1)
        ours, ours_mean = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        # monai = metrics.dice_score_monai(image1, image2)

        self.assertEqual([ours['1'], ours['2']], expected_scores)
        self.assertEqual(l2r, expected_scores)
        # self.assertEqual(monai, expected_scores)

    def test_dice_score_partial_overlap(self):
        """Test case where volumes have partial overlap."""
        image1 = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 0, 1], [1, 0, 0]]
        ]).to(torch.uint8)

        image2 = torch.tensor([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 1]]
        ]).to(torch.uint8)

        # we have 2 intersections and in both the top and bottom we have 4 values, so 2 * 2 / (4+4) = 0.5

        expected_scores = [0.5]  # Dice score should be 0.5
        ours, ours_mean = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        # monai = metrics.dice_score_monai(image1, image2)

        self.assertEqual([ours['1']], expected_scores)
        self.assertEqual(l2r, expected_scores)
        # self.assertEqual(monai, expected_scores)

    def test_dice_score_unequal_classes(self):
        image1 = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 2]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
        ]).to(torch.uint8)

        image2 = torch.tensor([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [3, 0, 1]]
        ]).to(torch.uint8)

        # we have 2 intersections and in both the top and bottom we have 4 values, so 2 * 2 / (4+4) = 0.5

        expected = [0.5, math.nan, math.nan]
        ours, ours_mean = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        # monai = metrics.dice_score_monai(image1, image2)

        self.assertTrue(compare_lists_with_nan([ours['1'], ours['2'], ours['3']],
                                               expected))
        self.assertTrue(compare_lists_with_nan(l2r,
                                               expected))
        # self.assertTrue(compare_lists_with_nan(monai,
        #                                        expected))

    def test_dice_score_no_zeros(self):
        image1 = torch.tensor([
            [[9, 9, 9], [9, 9, 9], [9, 9, 2]],
            [[9, 9, 9], [9, 1, 9], [9, 9, 1]],
            [[1, 9, 9], [9, 9, 1], [9, 9, 9]]
        ]).to(torch.uint8)

        image2 = torch.tensor([
            [[9, 1, 9], [9, 9, 9], [9, 9, 9]],
            [[9, 9, 9], [9, 1, 9], [9, 9, 9]],
            [[9, 9, 9], [9, 9, 1], [3, 9, 1]]
        ]).to(torch.uint8)

        expected = [0.5, math.nan, math.nan, 0.8636363636363636]
        ours, ours_mean = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        # monai = metrics.dice_score_monai(image1, image2)

        self.assertTrue(compare_lists_with_nan([ours['1'], ours['2'], ours['3'], ours['9']],
                                               expected))
        self.assertTrue(compare_lists_with_nan(l2r,
                                               expected))
        # self.assertTrue(compare_lists_with_nan(monai,
        #                                        expected))


class TestHausdorff(unittest.TestCase):

    def test_hd95_zero_dist(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        fixed = torch.tensor([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]).to(torch.uint8)

        warped = torch.tensor([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]).to(torch.uint8)

        expected = [0.0]

        res_l2r = metrics.hausdorff_distance_learn2reg(fixed,
                                                       warped, 0.95)
        res_monai, res_monai_mean = metrics.hausdorff_distance_monai(fixed,
                                                                     warped, 0.95)

        self.assertEqual(res_l2r, expected)
        self.assertEqual([res_monai['1']], expected)

    def test_hd95_one_dist(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        fixed = torch.tensor([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]).to(torch.uint8)

        warped = torch.tensor([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]).to(torch.uint8)

        expected = [1.0]

        res_l2r = metrics.hausdorff_distance_learn2reg(fixed,
                                                       warped,
                                                       1)
        res_monai, res_monai_mean = metrics.hausdorff_distance_monai(fixed,
                                                                     warped, 1)

        # self.assertEqual(res_l2r, expected)
        self.assertEqual([res_monai['1']], expected)

    def test_hd95_two_dist(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        fixed = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ]).to(torch.uint8)

        warped = torch.tensor([
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]],
            [[0, 0, 0], [1, 0, 0],  [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]]
        ]).to(torch.uint8)

        expected = [2.0]

        res_l2r = metrics.hausdorff_distance_learn2reg(fixed,
                                                       warped, 1)
        res_monai, res_monai_mean = metrics.hausdorff_distance_monai(fixed,
                                                                     warped, 1)

        # self.assertEqual(res_l2r, expected)
        self.assertEqual([res_monai['1']], expected)

    def test_hd95_diagonal_dist(self) -> None:
        """
        diagonal dist of sqrt(2)
        """

        fixed = torch.tensor([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]).to(torch.uint8)

        warped = torch.tensor([
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 1],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]).to(torch.uint8)

        expected = [math.sqrt(2.0)]

        res_l2r = metrics.hausdorff_distance_learn2reg(fixed,
                                                       warped, 1)
        res_monai, res_monai_mean = metrics.hausdorff_distance_monai(fixed,
                                                                     warped, 1)

        # self.assertEqual(res_ours, res_l2r[1])
        # self.assertEqual(res_l2r, res_monai)
        # self.assertTrue(compare_lists_with_nan(res_l2r, expected))
        self.assertTrue(compare_lists_with_nan([res_monai['1']], expected))

    def test_hd95_diagonal_and_horizontal_dist(self) -> None:
        """
        list of two distances : 1 and sqrt(2)
        """

        fixed = torch.tensor([
            [[0, 0, 0],
             [0, 2, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]).to(torch.uint8)

        warped = torch.tensor([
            [[0, 2, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 1],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ]).to(torch.uint8)

        expected = [math.sqrt(2.0), 1]

        # res_l2r = metrics.hausdorff_distance_learn2reg(fixed,
        #                                                warped, 1)
        res_monai, res_monai_mean = metrics.hausdorff_distance_monai(fixed,
                                                                     warped, 1)

        # self.assertEqual(res_ours, res_l2r[1])
        # self.assertEqual(res_l2r, res_monai)
        # self.assertTrue(compare_lists_with_nan(res_l2r, expected))
        self.assertTrue(compare_lists_with_nan(
            [res_monai['1'], res_monai['2']], expected))


class TestDisplacementFieldMetrics(unittest.TestCase):

    def test_jacobian_determinant(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        disp = torch.tensor([
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]).unsqueeze(-1).repeat(1, 1, 1, 3).float()

        print(disp.shape)

        ours = metrics.jacobian_determinant_from_displacement(disp)
        res_l2r = metrics.jacobian_determinant_from_displacement_l2r(disp)
        # res_monai = metrics.jacobian_determinant_from_displacement_monai(disp)
        print("expected: ?, l2r: ", res_l2r, "ours: ", ours)

        # TODO known negative jac dets
        # todo known ones e.g. pure translation

    def test_num_foldings(self):
        pass  # TODO

    def test_jacdet_visualization(self):
        pass  # TODO
