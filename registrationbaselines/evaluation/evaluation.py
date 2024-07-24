from pathlib import Path
import warnings

from typing import Optional

from tqdm import tqdm
import wandb
import numpy as np

from registrationbaselines.core import utils, result_csv
from registrationbaselines.core import metrics
from registrationbaselines.core import visualization
from registrationbaselines.data_loading.data_loaders import BaselineTransformations, GenericDataset


class Evaluation():
    """
    Class for evaluation of registration methods.

    It requires a precomputed transformation.
    """

    def __init__(self,
                 result_path: Path,
                 method: str,
                 dataset_data: GenericDataset,
                 dataset_transformations: BaselineTransformations) -> None:
        """
        Initialize the evaluatin model.
        """

        # create the csv file and all its parents if doesn't exist
        self.path_results = result_path / dataset_data.name / method / 'results.csv'
        self.path_results_plots = result_path / \
            dataset_data.name / method / 'results.pdf'
        self.path_plots = result_path / dataset_data.name / method / 'plots'
        self.path_results.parent.mkdir(parents=True, exist_ok=True)
        self.path_plots.mkdir(parents=True, exist_ok=True)
        self.path_results.touch()

        self.results = result_csv.EvaluationResults(
            self.path_results.as_posix())

        self.dataset_transformations = dataset_transformations
        self.dataset_data = dataset_data

    def evaluate(self) -> None:
        """
        Evaluate the registration model.
        """

        # assert len(dataset_transformations) == len(
        #     dataset_data), "Number of transformations and data must be the same."
        length_datasets = len(self.dataset_transformations)
        self.results.number_of_images = length_datasets

        for i in tqdm(range(length_datasets)):
            path_displacement = self.dataset_transformations[i]
            item = self.dataset_data[i]

            fixed_name = str(item["fixed_image"].stem).split('.')[0]

            self._evaluate_displacement(path_displacement, fixed_name)

            if self.dataset_data.has_segmentations:
                path_fixed = item["fixed_segmentations"]
                path_moving = item["moving_segmentations"]

            self._evaluate_segmentation(path_displacement,
                                        path_fixed,
                                        path_moving,
                                        fixed_name)

            if self.dataset_data.has_keypoints:
                path_fixed_keypoints = item["fixed_keypoints"]
                path_moving_keypoints = item["moving_keypoints"]

                self._evaluate_keypoints(path_displacement,
                                         path_fixed_keypoints,
                                         path_moving_keypoints,
                                         fixed_name)

        self.results.calculate_mean()
        self.results.calculate_stddev()
        self.results.calculate_min()
        self.results.calculate_max()

        self.results.write()
        self.results.plot(self.path_results_plots)

    def visualize(self,
                  idxs: Optional[list[int]] = None,
                  plot_to_wandb: Optional[bool] = False) -> None:
        """
        Create plots for the evaluation.
        """

        # TODO this should be removed once we worke with entire datasets
        warnings.warn("Restore the assert, when working with entire datasets.")
        # assert len(dataset_transformations) == len(
        #     dataset_data), "Number of transformations and data must be the same."

        if idxs is None:
            idxs = range(len(self.dataset_transformations))

        for i in tqdm(idxs):
            path_displacement = self.dataset_transformations[i]
            item = self.dataset_data[i]

            fixed_image_path = item["fixed_image"]
            moving_image_path = item["moving_image"]

            fixed_image = utils.load_image(fixed_image_path)
            moving_image = utils.load_image(moving_image_path)
            displacement = utils.load_displacement(path_displacement)

            deformed_image_path = self._get_deformed_image_path(fixed_image_path.name,
                                                                moving_image_path.name,
                                                                extension_overwrite=''.join(path_displacement.suffixes))
            deformed_image = utils.load_image(deformed_image_path)

            plots_path = self._create_plots_paths(fixed_image_path.name,
                                                  moving_image_path.name,
                                                  extension_overwrite=''.join(path_displacement.suffixes))

            fixed_keypoints = None
            moving_keypoints = None
            deformed_keypoints = None
            fixed_segmentation = None
            deformed_segmentation = None

            if self.dataset_data.has_segmentations:
                path_segmentation_fixed = item["fixed_segmentations"]
                path_segmentation_moving = item["moving_segmentations"]
                fixed_segmentation = utils.load_image(path_segmentation_fixed)
                moving_segmentation = utils.load_image(
                    path_segmentation_moving)

                deformed_segmentation = utils.deform_image(moving_segmentation,
                                                           displacement, mode='nearest')

            fixed_image = fixed_image.to(displacement.device)
            moving_image = moving_image.to(displacement.device)

            if self.dataset_data.has_keypoints:
                path_fixed_keypoints = item["fixed_keypoints"]
                path_moving_keypoints = item["moving_keypoints"]

                fixed_keypoints, moving_keypoints = metrics.read_lanmdarks(
                    path_fixed_keypoints, path_moving_keypoints)

                assert moving_keypoints.shape == moving_keypoints.shape
                assert fixed_keypoints.shape[-1] == 3 or fixed_keypoints.shape[-1] == 2

                deformed_keypoints = utils.deform_keypoints(
                    moving_keypoints, displacement)
            visualization.plot_all_registration_results(plots_path, moving_image.numpy(), fixed_image.numpy(), deformed_image.numpy(),
                                                        displacement.numpy(), fixed_labels=fixed_segmentation.numpy(), pred_labels=deformed_segmentation.numpy(),
                                                        fixed_keypoints=fixed_keypoints, moving_keypoints=moving_keypoints, pred_keypoints=deformed_keypoints)

    def _evaluate_displacement(self, path_displacement: Path, name: str) -> None:
        """
        Evaluates the displacement field with sdlogj and fraction of foldings.

        @param path_displacement: Path to the displacement field (torch tensor or nifti file).
        @param name: Name of the evaluated file pair
        """

        if not path_displacement.exists():
            raise FileNotFoundError(
                f"File {path_displacement} does not exist.")

        suffixes = path_displacement.suffixes
        if not suffixes == [".nii"] and \
                not suffixes == [".nii", ".gz"]:
            raise ValueError(
                f"Displacement file should have suffixes  '.nii' or '.nii.gz' but has {suffixes}.")

        displacement = utils.load_displacement(path_displacement)

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

        for path in [path_displacement, path_segmentation_fixed, path_segmentation_moving]:
            utils.is_nifti_and_exists(path)

        displacement = utils.load_displacement(path_displacement)
        segmentation_fixed = utils.load_image(path_segmentation_fixed)
        segmentation_moving = utils.load_image(path_segmentation_moving)

        warped = utils.deform_image(segmentation_moving,
                                    displacement,
                                    mode='nearest')

        deformed_segmentation_path = self._get_deformed_image_path(path_segmentation_fixed.name,
                                                                   path_segmentation_moving.name,
                                                                   extension_overwrite=''.join(path_displacement.suffixes))
        deformed_segmentation_path = Path(
            deformed_segmentation_path.as_posix().replace(".nii", "_seg.nii"))

        utils.save_image(warped, deformed_segmentation_path,
                         spacing=self.dataset_data.spacing)

        dice_scores = metrics.dice_score(
            segmentation_fixed, warped)

        if len(dice_scores) == 1:
            self.results.add_value("dice", dice_scores[0], name)
        else:
            for i, score in enumerate(dice_scores):
                self.results.add_value("dice_" + str(i), score, name)
                dice_mean += score

            dice_mean /= len(dice_scores)
            self.results.add_value("dice_mean", dice_mean, name)

        hausdorff_scores = metrics.hausdorff_distance(segmentation_fixed.squeeze(),
                                                      warped.squeeze())
        hausdorff95_scores = metrics.hausdorff_distance(segmentation_fixed.squeeze(),
                                                        warped.squeeze(),
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

    def _evaluate_keypoints(self,
                            path_displacement: Path,
                            path_fixed_keypoints: Path,
                            path_moving_keypoints: Path,
                            name: str) -> None:

        assert self.dataset_data is not None

        for path in [path_displacement, path_fixed_keypoints, path_moving_keypoints]:
            if not path.exists():
                raise FileNotFoundError(
                    f"File {path.as_posix()} does not exist.")

        displacement = utils.load_displacement(path_displacement)
        keypoints_fixed, keypoints_moving = metrics.read_lanmdarks(path_fixed_keypoints,
                                                                   path_moving_keypoints)

        keypoints_moving_warped = utils.deform_keypoints(keypoints_moving,
                                                         displacement.detach().cpu().numpy())

        warped_keypoints_path = self._get_deformed_image_path(path_fixed_keypoints.name,
                                                              path_moving_keypoints.name,
                                                              extension_overwrite="")
        warped_keypoints_path = Path(
            warped_keypoints_path.as_posix().replace(".csv", "") + ".csv")
        np.savetxt(warped_keypoints_path,
                   keypoints_moving_warped, delimiter=',')

        tre = metrics.tre(keypoints_fixed,
                          keypoints_moving,
                          keypoints_moving_warped,
                          self.dataset_data.spacing)
        tre30 = metrics.tre(keypoints_fixed,
                            keypoints_moving,
                            keypoints_moving_warped,
                            self.dataset_data.spacing,
                            percentile=30)

        self.results.add_value("tre", tre, name)
        self.results.add_value("tre30", tre30, name)

    def _create_plots_paths(self, name_fixed: str, name_moving: str, extension_overwrite=None):
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
        elif name_fixed.endswith(".pt"):
            name_fixed = name_fixed.replace(".pt", "")
            name_moving = name_moving.replace(".pt", "")

        path_plots = self.path_plots / \
            f"{name_moving}_deformed_to_{name_fixed}.pdf"

        return path_plots

    def _get_deformed_image_path(self, name_fixed: str, name_moving: str, extension_overwrite: str = "") -> Path:
        """
        Get the corresponding deformed image path.
        """

        extension = ""

        if name_fixed.endswith(".nii") or name_fixed.endswith(".nii.gz"):
            name_fixed = name_fixed.replace(".nii", "")
            name_moving = name_moving.replace(".nii", "")
            name_fixed = name_fixed.replace(".gz", "")
            name_moving = name_moving.replace(".gz", "")
            extension = ".nii.gz"

        elif name_fixed.endswith(".jpg"):
            name_fixed = name_fixed.replace(".jpg", "")
            name_moving = name_moving.replace(".jpg", "")

            extension = ".jpg"

        elif name_fixed.endswith(".pt"):
            name_fixed = name_fixed.replace(".pt", "")
            name_moving = name_moving.replace(".pt", "")
            extension = ".pt"

        if extension_overwrite != extension:
            extension = extension_overwrite

        path_plots: Path = self.path_results.parent / \
            f"deformed/{name_moving}_deformed_to_{name_fixed}{extension}"

        return path_plots

    def wandb_log(self):
        """
        Log the results to wandb.
        """
        # log quantitative results

        for key, value in self.results.df.loc["mean"].to_dict().items():
            if 'mean' not in key:
                new_key = key + "_mean"
            else:
                new_key = key.replace("_mean", "_overal_mean")

            wandb.log({new_key: float(value)})

        for key, value in self.results.df.loc["min"].to_dict().items():
            new_key = key + "_min"

            if 'mean' in key:
                new_key = key.replace("_mean", "_overal_min")

            wandb.log({new_key: float(value)})

        for key, value in self.results.df.loc["max"].to_dict().items():
            new_key = key + "_max"

            if 'mean' in key:
                new_key = key.replace("_mean", "_overal_max")

            wandb.log({new_key: float(value)})

        for key, value in self.results.df.loc["std"].to_dict().items():
            new_key = key + "_std"

            if 'mean' in key:
                new_key = key.replace("_mean", "_overal_std")

            wandb.log({new_key: float(value)})
