import math
import unittest
from pathlib import Path
import sys

import torch


sys.path.append(str(Path(__file__).parent.absolute().parent.parent))  # nopep8

import registrationbaselines.metrics.metrics as metrics


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
        ours = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        monai = metrics.dice_score_monai(image1.unsqueeze(0).unsqueeze(0), image2.unsqueeze(0).unsqueeze(0))

        print("expected: [1.0], l2r: ", l2r, "monai: ", monai, "ours: ", ours)

        self.assertEqual(ours, expected_scores)
        self.assertEqual(l2r, expected_scores)
        self.assertEqual(monai, expected_scores)

    # def test_dice_score_ignore_zero_class(self):
    #     """Test case where both volumes are identical, so Dice score should be 1."""
    #     image1 = torch.tensor([
    #         [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
    #         [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
    #         [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    #     ])
    #
    #     image2 = image1.clone()  # Identical to image1
    #
    #     len_expected_scores = 1  # One score - we ignore 0 class
    #     actual_scores = metrics.dice_score(image1, image2)
    #     self.assertEqual(len(actual_scores), len_expected_scores)

    def test_dice_score_no_overlap(self):
        """Test case where volumes have no overlap, so Dice score should be 0."""
        image1 = torch.tensor([
            [[2, 2, 2], [1, 1, 1], [2, 2, 2]],
            [[2, 2, 2], [1, 1, 1], [2, 2, 2]],
            [[2, 2, 2], [1, 1, 1], [2, 2, 2]]
        ])

        image2 = torch.tensor([
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]],
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]],
            [[1, 1, 1], [2, 2, 2], [1, 1, 1]]
        ])

        expected_scores = [0.0, 0.0]  # One score for each class (1 and 2)
        ours = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        monai = metrics.dice_score_monai(image1, image2)

        print("expected: [0.0, 0.0], l2r: ", l2r, "monai: ", monai, "ours: ", ours)

        self.assertEqual(ours, expected_scores)
        self.assertEqual(l2r, expected_scores)
        self.assertEqual(monai, expected_scores)

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
        ours = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        monai = metrics.dice_score_monai(image1, image2)

        print("expected: [0.0, 0.0], l2r: ", l2r, "monai: ", monai, "ours: ", ours)

        self.assertEqual(ours, expected_scores)
        self.assertEqual(l2r, expected_scores)
        self.assertEqual(monai, expected_scores)

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
        ours = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        monai = metrics.dice_score_monai(image1, image2)

        print("expected: [0.5], l2r: ", l2r, "monai: ", monai, "ours: ", ours)

        self.assertEqual(ours, expected_scores)
        self.assertEqual(l2r, expected_scores)
        self.assertEqual(monai, expected_scores)

    def test_dice_score_unequal_classes(self):
        image1 = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 2]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 0, 1], [3, 0, 0]]
        ])

        image2 = torch.tensor([
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [3, 0, 1]]
        ])

        # we have 2 intersections and in both the top and bottom we have 4 values, so 2 * 2 / (4+4) = 0.5

        expected_scores = [0.5]  # Dice score should be 0.5
        ours = metrics.dice_score(image1, image2)
        l2r = metrics.dice_score_l2r(image1, image2, image2)
        monai = metrics.dice_score_monai(image1, image2)

        print("expected: [0.5], l2r: ", l2r, "monai: ", monai, "ours: ", ours)

        self.assertEqual(ours, monai)
        self.assertEqual(l2r, monai)


class TestHausdorff(unittest.TestCase):

    def test_hd95_zero_dist(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        fixed = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])

        warped = torch.tensor([
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0],  [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]]
        ])

        res_l2r = metrics.hausdorff_distance_learn2reg(fixed,
                                              warped, 0.95)
        res_monai = metrics.hausdorff_distance_monai(fixed.unsqueeze(0),
                                              warped.unsqueeze(0), 0.95)
        print("expected: 0, l2r: ",res_l2r, "monai: ", res_monai)

        # self.assertEqual(res_ours, res_l2r[1])
        self.assertEqual(res_l2r, res_monai)
        self.assertEqual(res_l2r, [0.0])

    def test_hd95_one_dist(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        fixed = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])

        warped = torch.tensor([
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1],  [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]]
        ])

        res_l2r = metrics.hausdorff_distance_learn2reg(fixed,
                                       warped,
                                       1)
        res_monai = metrics.hausdorff_distance_monai(fixed.unsqueeze(0),
                                                     warped.unsqueeze(0), 1)

        print("expected: 1, l2r: ", res_l2r, "monai: ", res_monai)

        self.assertEqual(res_l2r, res_l2r[0])
        self.assertEqual(res_l2r, res_monai)
        self.assertEqual(res_l2r, [1.0])

    def test_hd95_two_dist(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        fixed = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])

        warped = torch.tensor([
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]],
            [[0, 0, 0], [1, 0, 0],  [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]]
        ])

        res_ours = metrics.hausdorff_distance_learn2reg(fixed,
                                              warped, 1)
        res_monai = metrics.hausdorff_distance_monai(fixed.unsqueeze(0),
                                              warped.unsqueeze(0), 1)
        print("expected: 2, l2r: ",res_ours, "monai: ", res_monai)

        # self.assertEqual(res_ours, res_l2r[1])
        self.assertEqual(res_ours, res_monai)
        self.assertEqual(res_ours, [2.0])


    def test_hd95_diagonal_dist(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        fixed = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])

        warped = torch.tensor([
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0],  [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0],  [0, 0, 0]]
        ])

        # res_l2r = metrics.compute_hd95(fixed.detach().cpu().numpy(),
        #                                fixed.detach().cpu().numpy(),
        #                                warped.detach().cpu().numpy(),
        #                                [1])
        res_l2r = metrics.hausdorff_distance_learn2reg(fixed,
                                              warped, 1)
        res_monai = metrics.hausdorff_distance_monai(fixed.unsqueeze(0),
                                              warped.unsqueeze(0), 1)
        print("expected: ", math.sqrt(2), "l2r: ",res_l2r, "monai: ", res_monai)

        # self.assertEqual(res_ours, res_l2r[1])
        self.assertEqual(res_l2r, res_monai)
        self.assertEqual(res_l2r, [math.sqrt(2.0)])


class TestDisplacementFieldMetrics(unittest.TestCase):

    def test_jacobian_determinant(self) -> None:
        """
        Test case where volumes are identical, so Hausdorff distance should be 0.
        """

        disp = torch.tensor([
            [[0, 0, 0,0], [0, 0, 0,0], [0, 0, 0,0],[0, 0, 0,0]],
            [[0, 0, 0,0], [0, 1, 0,0], [0, 0, 0,0],[0, 0, 0,0]],
            [[0, 0, 0,0], [0, 0, 0,0], [0, 0, 0,0],[0, 0, 0,0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0,0]],
        ]).unsqueeze(-1).repeat(1, 1, 1, 3).float()

        print(disp.shape)

        ours = metrics.jacobian_determinant_from_displacement(disp)
        res_l2r = metrics.jacobian_determinant_from_displacement_l2r(disp)
        # res_monai = metrics.jacobian_determinant_from_displacement_monai(disp)
        print("expected: ?, l2r: ",res_l2r, "ours: ", ours)

        # TODO known negative jac dets
        # todo known ones e.g. pure translation

    def test_num_foldings(self):
        pass #TODO

    def test_jacdet_visualization(self):
        pass # TODO