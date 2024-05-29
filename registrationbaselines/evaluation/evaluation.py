from pathlib import Path
import shutil
import warnings
from typing import Optional

from tqdm import tqdm
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from registrationbaselines.core import utils, result_csv, utils_metrics
from registrationbaselines.transforms import general_deformation
from registrationbaselines.core import metrics
from registrationbaselines.core import visualization


class Evaluation():
    """
    Class for evaluation of registration methods.

    It requires a precomputed transformation.
    """

    def __init__(self, configuration_path: Path):
        """
        Initialize the registration model.
        """

        self.configuration_path = configuration_path
        self.configuration = utils.read_config(configuration_path)

        method = self.configuration['method_name']

        # create the csv file and all its parents if doesn't exist
        self.path_results = Path(
            self.configuration['result_path']) / method / 'results.csv'
        self.path_results_plots = Path(
            self.configuration['result_path']) / method / 'results.pdf'
        self.path_plots = Path(
            self.configuration['result_path']) / method / 'plots'
        self.path_results.parent.mkdir(parents=True, exist_ok=True)
        self.path_plots.mkdir(parents=True, exist_ok=True)
        self.path_results.touch()

        self.results = result_csv.EvaluationResults(self.path_results)

        # get the transformaton class based on the method name
        self.transformation = general_deformation.GeneralDeformation(
            self.configuration["result_path"])

        self.transformation.path_deformed = Path(
            self.configuration['result_path']) / "temp"

        self.path_warped = None

    def __del__(self):
        """
        Clean up the temporary directory (called at the end of scope to delete the temporary directory).
        """

        if self.transformation.path_deformed is not None and self.transformation.path_deformed.exists():
            shutil.rmtree(self.transformation.path_deformed)

    def evaluate(self, dataset_transformations: Dataset, dataset_data: Dataset) -> None:
        """
        Evaluate the registration model.
        """

        # TODO this should be removed once we worke with entire datasets
        warnings.warn("Restore the assert, when working with entire datasets.")
        # assert len(dataset_transformations) == len(
        #     dataset_data), "Number of transformations and data must be the same."
        length_datasets = len(dataset_transformations)
        self.results.number_of_images = length_datasets

        self.dataset_data = dataset_data

        for i in tqdm(range(length_datasets)):
            path_transformation = dataset_transformations[i]
            item = dataset_data[i]

            fixed_name = str(item["images"][0].stem).split('.')[0]

            self._evaluate_displacement(path_transformation, fixed_name)

            if "segmentations" in item:
                path_moving = item["segmentations"][0]
                path_fixed = item["segmentations"][1]

                self._evaluate_segmentation(path_transformation,
                                            path_fixed,
                                            path_moving,
                                            fixed_name)

            if "landmarks" in item:
                path_moving_landmarks = item["landmarks"][0]
                path_fixed_landmarks = item["landmarks"][1]

                self._evaluate_landmarks(path_transformation,
                                         path_fixed_landmarks,
                                         path_moving_landmarks,
                                         fixed_name)

        self.results.calculate_mean()
        self.results.calculate_stddev()
        self.results.calculate_min()
        self.results.calculate_max()

        self.results.write()
        self.results.plot(self.path_results_plots)

    def visualize(self, dataset_transformations: Dataset, dataset_data: Dataset, idxs: Optional[list[int]] = None,
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

        for i in tqdm(idxs):
            path_transformation = dataset_transformations[i]
            item = dataset_data[i]
            moving_image_path = item["images"][0]
            fixed_image_path = item["images"][1]
            moving_image = nib.load(moving_image_path).get_fdata()
            fixed_image = nib.load(fixed_image_path).get_fdata()
            displacement = nib.load(
                path_transformation.as_posix()).get_fdata().squeeze()
            deformed_image_path = self._get_deformed_image_path(
                moving_image_path.name, fixed_image_path.name)
            deformed_image = nib.load(deformed_image_path).get_fdata()

            plots_path = self._create_plots_paths(
                fixed_image_path.name, moving_image_path.name)

            if "segmentations" in item:
                moving_segmentation = nib.load(
                    item["segmentations"][0]).get_fdata()
                fixed_segmentation = nib.load(
                    item["segmentations"][1]).get_fdata()
                # deformed_segmentation = utils_metrics.deform_segmentations(moving_segmentation, displacement)
                deformed_segmentation = None  # TODO implement function above
            if "landmarks" in item:
                moving_landmarks = np.genfromtxt(
                    item["landmarks"][0], delimiter=',')
                fixed_landmarks = np.genfromtxt(
                    item["landmarks"][1], delimiter=',')
                deformed_landmarks = utils_metrics.deform_landmarks(
                    moving_landmarks, displacement)
            visualization.plot_all_registration_results(plots_path, moving_image, fixed_image, deformed_image,
                                                        displacement, fixed_segmentation, deformed_segmentation,
                                                        fixed_landmarks, moving_landmarks, deformed_landmarks)

    def _evaluate_displacement(self, path_displacement: Path, name: str) -> None:

        sd_log_det, fraction_foldings = metrics.displacement_field_metrics(
            path_displacement)

        self.results.add_value("sdlogj", sd_log_det, name)
        self.results.add_value("frac_foldings", fraction_foldings, name)

    def _evaluate_segmentation(self,
                               path_transformation: Path,
                               path_fixed_segmentation: Path,
                               path_moving_segmentation: Path,
                               name: str) -> None:
        """
        Evaluate segmentations.

        If there is only one class it is trivial. If there are more classes, we have to
        create new temporary segmentation files for each class and evaluate them
        (because of interpolation issues).

        Args:

            path_transformation (Path): The path to the transformation file.
            path_fixed_segmentation (Path): The path to the fixed segmentation file.
            path_moving_segmentation (Path): The path to the moving segmentation file.
            name (str): The name of the evaluation.

        Returns:
            None
        """

        dice_mean = 0
        hausdorff_mean = 0
        hausdorff95_mean = 0

        # create temp directory
        temp_dir = Path(
            self.configuration['result_path']) / self.configuration['method_name'] / "temp"
        temp_dir.mkdir(exist_ok=True)

        self.path_warped = self.transformation.apply_transformation(path_fixed_segmentation,
                                                                    path_moving_segmentation,
                                                                    path_transformation,
                                                                    temp_dir / "temp_defrmed.nii.gz",
                                                                    sitk.sitkNearestNeighbor)

        dice_scores = metrics.dice_score(path_fixed_segmentation,
                                         self.path_warped)
        if len(dice_scores) == 1:
            self.results.add_value("dice", dice_scores[0], name)
        else:
            for i, score in enumerate(dice_scores):
                self.results.add_value("dice_" + str(i), score, name)
                dice_mean += score

            dice_mean /= len(dice_scores)
            self.results.add_value("dice_mean", dice_mean, name)

        hausdorff_scores = metrics.hausdorff_distance(path_fixed_segmentation,
                                                      self.path_warped)
        hausdorff95_scores = metrics.hausdorff_distance(path_fixed_segmentation,
                                                        self.path_warped,
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

        # delete all files in the temp directory
        shutil.rmtree(temp_dir)

    def _evaluate_landmarks(self,
                            path_transformation: Path,
                            path_fixed_landmarks: Path,
                            path_moving_landmarks: Path,
                            name: str) -> None:

        assert self.dataset_data is not None

        tre = metrics.tre(path_fixed_landmarks, path_moving_landmarks,
                          path_transformation, self.dataset_data.spacing)
        tre30 = metrics.tre(path_fixed_landmarks, path_moving_landmarks, path_transformation, self.dataset_data.spacing,
                            percentile=30)

        self.results.add_value("tre", tre, name)
        self.results.add_value("tre30", tre30, name)

    def _create_plots_paths(self, name_fixed: str, name_moving: str):
        """
        Create the paths for the plots.
        """

        name_moving = name_moving.replace(".nii", "")
        name_fixed = name_fixed.replace(".nii", "")

        name_moving = name_moving.replace(".gz", "")
        name_fixed = name_fixed.replace(".gz", "")

        path_plots = self.path_plots / \
            f"{name_moving}_deformed_to_{name_fixed}.pdf"
        path_plots = path_plots.resolve().as_posix()

        return Path(path_plots)

    def _get_deformed_image_path(self, name_fixed: str, name_moving: str):
        """
        Get the corresponding deformed image path.
        """

        name_moving = name_moving.replace(".nii", "")
        name_fixed = name_fixed.replace(".nii", "")
        name_moving = name_moving.replace(".gz", "")
        name_fixed = name_fixed.replace(".gz", "")

        path_plots = self.path_results.parent / \
            f"deformed/{name_moving}_deformed_to_{name_fixed}.nii"
        path_plots = path_plots.resolve().as_posix()

        return Path(path_plots)
