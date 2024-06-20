from pathlib import Path
import shutil
import warnings
from typing import Optional

from tqdm import tqdm
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from registrationbaselines.core import utils, result_csv, general_deformation
from registrationbaselines.core import metrics
from registrationbaselines.core import visualization
from registrationbaselines.data_loading.data_loaders import BaselineTransformations


class Evaluation():
    """
    Class for evaluation of registration methods.

    It requires a precomputed transformation.
    """

    def __init__(self, result_path: Path, method: str) -> None:
        """
        Initialize the evaluatin model.
        """

        # create the csv file and all its parents if doesn't exist
        self.path_results = result_path / method / 'results.csv'
        self.path_results_plots = result_path / method / 'results.pdf'
        self.path_plots = result_path / method / 'plots'
        self.path_results.parent.mkdir(parents=True, exist_ok=True)
        self.path_plots.mkdir(parents=True, exist_ok=True)
        self.path_results.touch()

        self.results = result_csv.EvaluationResults(self.path_results)

    def evaluate(self, dataset_transformations: BaselineTransformations, dataset_data: Dataset) -> None:
        """
        Evaluate the registration model.
        """

        # assert len(dataset_transformations) == len(
        #     dataset_data), "Number of transformations and data must be the same."
        length_datasets = len(dataset_transformations)
        self.results.number_of_images = length_datasets

        self.dataset_data = dataset_data
        for i in tqdm(range(length_datasets)):
            path_displacement = dataset_transformations[i]
            item = dataset_data[i]

            fixed_name = str(item["images"][0].stem).split('.')[0]

            self._evaluate_displacement(path_displacement, fixed_name)

            if "segmentations" in item:
                path_fixed = item["segmentations"][0]
                path_moving = item["segmentations"][1]

                self._evaluate_segmentation(path_displacement,
                                            path_fixed,
                                            path_moving,
                                            fixed_name)

            if "landmarks" in item:
                # is2d = sitk.GetArrayFromImage(sitk.ReadImage(
                #     path_displacement, sitk.sitkVectorFloat64)).shape[-1] == 2

                if dataset_data.ndim == 2:
                    path_fixed_landmarks = item["landmarks"]
                    path_moving_landmarks = item["landmarks"]
                else:
                    path_fixed_landmarks = item["landmarks"][0]
                    path_moving_landmarks = item["landmarks"][1]

                self._evaluate_landmarks(path_displacement,
                                         path_fixed_landmarks,
                                         path_moving_landmarks,
                                         fixed_name)

        self.results.calculate_mean()
        self.results.calculate_stddev()
        self.results.calculate_min()
        self.results.calculate_max()

        self.results.write()
        self.results.plot(self.path_results_plots)

    def visualize(self, dataset_transformations: BaselineTransformations, dataset_data: Dataset, idxs: Optional[list[int]] = None,
                  plot_to_wandb: Optional[bool] = False) -> None:
        """
        Create plots for the evaluation.
        """

        # TODO this should be removed once we worke with entire datasets
        warnings.warn("Restore the assert, when working with entire datasets.")
        # assert len(dataset_transformations) == len(
        #     dataset_data), "Number of transformations and data must be the same."

        if idxs is None:
            idxs = range(len(dataset_transformations))

        print(idxs)

        for i in tqdm(idxs):
            print(i)
            path_displacement = dataset_transformations[i]
            item = dataset_data[i]
            fixed_image_path = item["images"][0]
            moving_image_path = item["images"][1]
            fixed_image = sitk.ReadImage(fixed_image_path)
            moving_image = sitk.ReadImage(moving_image_path)
            # displacement = sitk.ReadImage(
            #     path_displacement.as_posix(), sitk.sitkVectorFloat64)
            displacement = nib.load(
                path_displacement.as_posix()).get_fdata().squeeze()
            deformed_image_path = self._get_deformed_image_path(
                fixed_image_path.name, moving_image_path.name)
            deformed_image = sitk.ReadImage(deformed_image_path)

            plots_path = self._create_plots_paths(
                fixed_image_path.name, moving_image_path.name)

            fixed_landmarks = None
            moving_landmarks = None
            deformed_landmarks = None
            fixed_segmentation = None
            deformed_segmentation = None

            if "segmentations" in item:
                fixed_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(
                    item["segmentations"][0]))
                # moving_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(
                #     item["segmentations"][1]))
                # deformed_segmentation = utils_metrics.deform_segmentations(moving_segmentation, displacement)
                deformed_segmentation = None  # TODO implement function above
            if "landmarks" in item:

                if dataset_data.ndim == 2:
                    path_fixed_landmarks = item["landmarks"]
                    path_moving_landmarks = item["landmarks"]
                else:
                    path_fixed_landmarks = item["landmarks"][0]
                    path_moving_landmarks = item["landmarks"][1]

                landmarks_fixed, landmarks_moving = metrics.read_lanmdarks(
                    path_fixed_landmarks, path_moving_landmarks)

                assert landmarks_moving.shape == landmarks_fixed.shape
                assert landmarks_fixed.shape[-1] == 3 or landmarks_fixed.shape[-1] == 2

                # fixed_landmarks = np.genfromtxt(path_fixed_landmarks, delimiter=',')
                # moving_landmarks = np.genfromtxt(path_moving_landmarks, delimiter=',')

                deformed_landmarks = utils.deform_landmarks(
                    landmarks_moving, displacement)
                # deformed_landmarks = utils_metrics.deform_landmarks(
                #     moving_landmarks, displacement)
            visualization.plot_all_registration_results(plots_path, sitk.GetArrayFromImage(moving_image), sitk.GetArrayFromImage(fixed_image), sitk.GetArrayFromImage(deformed_image),
                                                        displacement, fixed_labels=fixed_segmentation, pred_labels=deformed_segmentation,
                                                        fixed_keypoints=landmarks_fixed, moving_keypoints=landmarks_moving, pred_keypoints=deformed_landmarks)

    def _evaluate_displacement(self, path_displacement: Path, name: str) -> None:

        displacement = sitk.ReadImage(
            path_displacement, sitk.sitkVectorFloat64)

        sd_log_det, fraction_foldings = metrics.displacement_field_metrics(
            displacement)

        self.results.add_value("sdlogj", sd_log_det, name)
        self.results.add_value("frac_foldings", fraction_foldings, name)

    def _evaluate_segmentation(self,
                               path_displacement: Path,
                               path_segmentation_fixed: Path,
                               path_segmentation_moving: Path,
                               name: str) -> None:
        """
        Evaluate segmentations.

        If there is only one class it is trivial. If there are more classes, we have to
        create new temporary segmentation files for each class and evaluate them
        (because of interpolation issues).

        Args:

            path_displacement (Path): The path to the transformation file.
            path_segmentation_fixed (Path): The path to the fixed segmentation file.
            path_segmentation_moving (Path): The path to the moving segmentation file.
            name (str): The name of the evaluation.

        Returns:
            None
        """

        dice_mean = 0
        hausdorff_mean = 0
        hausdorff95_mean = 0

        displacement = sitk.ReadImage(
            path_displacement, sitk.sitkVectorFloat64)
        segmentation_fixed = sitk.ReadImage(path_segmentation_fixed)
        segmentation_moving = sitk.ReadImage(path_segmentation_moving)

        warped = utils.deform_image(segmentation_fixed,
                                    segmentation_moving,
                                    displacement,
                                    sitk.sitkNearestNeighbor)

        dice_scores = metrics.dice_score(segmentation_fixed, warped)
        if len(dice_scores) == 1:
            self.results.add_value("dice", dice_scores[0], name)
        else:
            for i, score in enumerate(dice_scores):
                self.results.add_value("dice_" + str(i), score, name)
                dice_mean += score

            dice_mean /= len(dice_scores)
            self.results.add_value("dice_mean", dice_mean, name)

        hausdorff_scores = metrics.hausdorff_distance(segmentation_fixed,
                                                      warped)
        hausdorff95_scores = metrics.hausdorff_distance(segmentation_fixed,
                                                        warped,
                                                        percentile=95)

        assert len(hausdorff_scores) == len(
            hausdorff95_scores), "Hausdorff scores and 95th percentile scores should have the same length."

        if len(hausdorff_scores) == 1:
            self.results.add_value("hausdorff", hausdorff_scores[0], name)
            self.results.add_value("hausdorff95", hausdorff95_scores[0], name)
        else:
            for i, score in enumerate(hausdorff_scores):
                self.results.add_value("hausdorff_" + str(i), score, name)
                hausdorff_mean += score

                self.results.add_value(
                    "hausdorff95_" + str(i), hausdorff95_scores[i], name)
                hausdorff95_mean += hausdorff95_scores[i]

            hausdorff_mean /= len(hausdorff_scores)
            self.results.add_value("hausdorff_mean", hausdorff_mean, name)

            hausdorff95_mean /= len(hausdorff95_scores)
            self.results.add_value("hausdorff95_mean", hausdorff95_mean, name)

    def _evaluate_landmarks(self,
                            path_displacement: Path,
                            path_fixed_landmarks: Path,
                            path_moving_landmarks: Path,
                            name: str) -> None:

        assert self.dataset_data is not None

        tre = metrics.tre(path_fixed_landmarks, path_moving_landmarks,
                          path_displacement, self.dataset_data.spacing)
        tre30 = metrics.tre(path_fixed_landmarks, path_moving_landmarks, path_displacement, self.dataset_data.spacing,
                            percentile=30)

        self.results.add_value("tre", tre, name)
        self.results.add_value("tre30", tre30, name)

    def _create_plots_paths(self, name_fixed: str, name_moving: str):
        """
        Create the paths for the plots.
        """

        if name_fixed.endswith(".nii") or name_fixed.endswith(".nii.gz"):
            name_fixed = name_fixed.replace(".nii", "")
            name_moving = name_moving.replace(".nii", "")

            name_fixed = name_fixed.replace(".gz", "")
            name_moving = name_moving.replace(".gz", "")
        elif name_fixed.endswith(".jpg"):
            name_fixed = name_fixed.replace(".jpg", "")
            name_moving = name_moving.replace(".jpg", "")
        path_plots = self.path_plots / \
            f"{name_moving}_deformed_to_{name_fixed}.pdf"
        path_plots = path_plots.resolve().as_posix()

        return Path(path_plots)

    def _get_deformed_image_path(self, name_fixed: str, name_moving: str):
        """
        Get the corresponding deformed image path.
        """

        if name_fixed.endswith(".nii") or name_fixed.endswith(".nii.gz"):
            name_fixed = name_fixed.replace(".nii", "")
            name_moving = name_moving.replace(".nii", "")
            name_fixed = name_fixed.replace(".gz", "")
            name_moving = name_moving.replace(".gz", "")
            path_plots = self.path_results.parent / \
                f"deformed/{name_moving}_deformed_to_{name_fixed}.nii.gz"
        elif name_fixed.endswith(".jpg"):
            name_fixed = name_fixed.replace(".jpg", "")
            name_moving = name_moving.replace(".jpg", "")

            path_plots = self.path_results.parent / \
                f"deformed/{name_moving}_deformed_to_{name_fixed}.jpg"
        path_plots = path_plots.resolve().as_posix()

        return Path(path_plots)
