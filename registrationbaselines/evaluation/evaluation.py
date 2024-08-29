from pathlib import Path

from typing import Optional, Union

from tqdm import tqdm
import wandb
import numpy as np
import torch

from registrationbaselines.evaluation import result_csv, plot_objects
from registrationbaselines.io import load, save
from registrationbaselines.core import utils
from registrationbaselines.warping import deform_objects
from registrationbaselines.metrics import metrics
from registrationbaselines.data_loading.data_loaders import BaselineTransformations, GenericDataset
from registrationbaselines.core.types import intArray3D


class Evaluation():
    """
    Class for evaluation of registration methods.

    It requires a precomputed transformation.
    """

    def __init__(self,
                 run_path: Path,
                 dataset_data: GenericDataset,
                 dataset_transformations: Optional[BaselineTransformations] = None,
                 use_zero_displacement: bool = False,
                 use_masked_evaluation: Optional[bool] = None) -> None:
        """
        Initialize the evaluatin model.
        """

        self.use_zero_displacement = use_zero_displacement
        self.use_masked_evaluation = use_masked_evaluation

        # create the csv file and all its parents if doesn't exist
        self.path_results = run_path / 'results.csv'
        self.path_results_plots = run_path / 'results.pdf'

        self.path_plots = run_path / 'plots'
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

        length_datasets = len(self.dataset_data)
        self.results.number_of_images = length_datasets

        for i in tqdm(range(length_datasets)):

            item = self.dataset_data[i]
            fixed_name = str(item["fixed_image"].stem).split('.')[0]

            if self.use_masked_evaluation:
                image_fixed = load.load_image(
                    item["fixed_image"])
                fixed_evaluation_mask: Union[intArray3D, None] = utils.get_convex_hull_mask(
                    image_fixed.detach().cpu().numpy())
            else:
                fixed_evaluation_mask = None

            if not self.use_zero_displacement:
                path_displacement = self.dataset_transformations[i]

                self._evaluate_displacement(path_displacement,
                                            fixed_name,
                                            fixed_evaluation_mask)

                if self.dataset_data.has_segmentations:
                    self._evaluate_segmentation(item["fixed_segmentations"],
                                                item["moving_segmentations"],
                                                fixed_name,
                                                path_displacement,
                                                fixed_evaluation_mask)
            else:
                if self.dataset_data.has_segmentations:
                    self._evaluate_segmentation(item["fixed_segmentations"],
                                                item["moving_segmentations"],
                                                fixed_name,
                                                None,
                                                fixed_evaluation_mask)

            """
            if self.dataset_data.has_keypoints:
                path_fixed_keypoints = item["fixed_keypoints"]
                path_moving_keypoints = item["moving_keypoints"]

                self._evaluate_keypoints(path_displacement,
                                         path_fixed_keypoints,
                                         path_moving_keypoints,
                                         fixed_name)
            """

        self.results.calculate_mean()
        self.results.calculate_stddev()
        self.results.calculate_min()
        self.results.calculate_max()

        self.results.write()
        self.results.plot(self.path_results_plots)

    def visualize(self,
                  idxs: Optional[list[int]] = None) -> None:
        """
        Create plots for the evaluation.
        """

        if idxs is None:
            idxs = range(len(self.dataset_data))

        for i in tqdm(idxs):
            item = self.dataset_data[i]

            fixed_image_path = item["fixed_image"]
            moving_image_path = item["moving_image"]

            fixed_image = load.load_image(fixed_image_path)
            moving_image = load.load_image(moving_image_path)

            if self.use_zero_displacement:
                shape = self.dataset_data.image_shape
                displacement = torch.zeros((*shape, 3), dtype=torch.float32)
                deformed_image = moving_image.detach().clone()
            else:
                path_displacement = self.dataset_transformations[i]
                displacement = load.load_displacement(path_displacement)

                deformed_image_path = self._get_deformed_image_path(fixed_image_path.name,
                                                                    moving_image_path.name,
                                                                    extension_overwrite=''.join(path_displacement.suffixes))
                # todo check that loaded is the same as deformed up to some epsilon
                # deformed_image = load.load_image(deformed_image_path)
                deformed_image = deform_objects.deform_image(moving_image,
                                                             displacement)

            if self.use_masked_evaluation:
                fixed_mask = utils.get_convex_hull_mask(
                    fixed_image.detach().cpu().numpy())
                displacement *= np.stack([fixed_mask] * 3, axis=-1)

            plots_path = self._create_plots_paths(fixed_image_path.name,
                                                  moving_image_path.name,
                                                  extension_overwrite='.nii.gz')

            fixed_keypoints = None
            moving_keypoints = None
            deformed_keypoints = None
            fixed_segmentation = None
            deformed_segmentation = None

            if self.dataset_data.has_segmentations:
                path_segmentation_fixed = item["fixed_segmentations"]
                path_segmentation_moving = item["moving_segmentations"]
                fixed_segmentation = load.load_image(path_segmentation_fixed)
                moving_segmentation = load.load_image(
                    path_segmentation_moving)

                deformed_segmentation = deform_objects.deform_image(moving_segmentation,
                                                                    displacement)

                if self.use_masked_evaluation:
                    # fixed_mask = utils_metrics.get_convex_hull_mask(
                    #     fixed_segmentation.detach().cpu().numpy())
                    deformed_segmentation *= fixed_mask

            fixed_image = fixed_image.to(displacement.device)
            moving_image = moving_image.to(displacement.device)

            if self.dataset_data.has_keypoints:
                path_fixed_keypoints = item["fixed_keypoints"]
                path_moving_keypoints = item["moving_keypoints"]

                fixed_keypoints = load.load_keypoints(path_fixed_keypoints)
                moving_keypoints = load.load_keypoints(path_moving_keypoints)

                assert moving_keypoints.shape == moving_keypoints.shape
                assert fixed_keypoints.shape[-1] == 3 or fixed_keypoints.shape[-1] == 2

                deformed_keypoints = deform_objects.deform_keypoints(
                    moving_keypoints, displacement)
            plot_objects.plot_all_registration_results(moving_image,
                                                       fixed_image,
                                                       deformed_image,
                                                       displacement,
                                                       fixed_segmentations=fixed_segmentation,
                                                       pred_segmentations=deformed_segmentation,
                                                       fixed_keypoints=fixed_keypoints,
                                                       moving_keypoints=moving_keypoints,
                                                       pred_keypoints=deformed_keypoints,
                                                       save_path=plots_path)

    def _evaluate_displacement(self,
                               path_displacement: Path,
                               name: str,
                               fixed_evaluation_mask: Optional[intArray3D] = None) -> None:
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

        displacement = load.load_displacement(path_displacement)

        sd_log_det, fraction_foldings = metrics.displacement_field_metrics(
            displacement,
            fixed_evaluation_mask)

        self.results.add_value("sdlogj", sd_log_det, name)
        self.results.add_value("frac_foldings", fraction_foldings, name)

    def _evaluate_segmentation(self,
                               path_segmentation_fixed: Path,
                               path_segmentation_moving: Path,
                               name: str,
                               path_displacement: Optional[Path] = None,
                               fixed_evaluation_mask: Optional[intArray3D] = None) -> None:
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

        utils.is_nifti(path_segmentation_fixed)
        utils.is_nifti(path_segmentation_moving)

        segmentation_fixed = load.load_image(path_segmentation_fixed)
        segmentation_moving = load.load_image(path_segmentation_moving)

        if not self.use_zero_displacement and path_displacement:
            utils.is_nifti(path_displacement)
            displacement = load.load_displacement(path_displacement)
            segmentation_warped = deform_objects.deform_image(segmentation_moving,
                                                              displacement)
            # save the deformed segmentation
            deformed_segmentation_path = self._get_deformed_image_path(path_segmentation_fixed.name,
                                                                       path_segmentation_moving.name,
                                                                       extension_overwrite='.nii.gz')
            deformed_segmentation_path = Path(
                deformed_segmentation_path.as_posix().replace(".nii", "_seg.nii"))
            save.save_image(segmentation_warped, deformed_segmentation_path,
                            spacing=self.dataset_data.spacing)

        elif self.use_zero_displacement and not path_displacement:
            shape = self.dataset_data.image_shape
            displacement = torch.zeros((*shape, 3), dtype=torch.float32)
            segmentation_warped = segmentation_moving.detach().clone()
        else:
            raise ValueError("Displacement field not found.")

        if fixed_evaluation_mask is not None:
            segmentation_warped *= fixed_evaluation_mask

        dice_scores, dice_mean = metrics.dice_score(segmentation_fixed,
                                                    segmentation_warped)

        if len(dice_scores) == 1:
            self.results.add_value("dice",
                                   next(iter(dice_scores.values())),
                                   name)
        else:
            for cls, score in dice_scores.items():
                self.results.add_value("dice_" + cls,
                                       score,
                                       name)

            self.results.add_value("dice_mean",
                                   dice_mean,
                                   name)

        hausdorff_scores, hausdorff_mean = metrics.hausdorff_distance_monai(segmentation_fixed.squeeze(),
                                                                            segmentation_warped.squeeze())
        hausdorff95_scores, hausdorff95_mean = metrics.hausdorff_distance_monai(segmentation_fixed.squeeze(),
                                                                                segmentation_warped.squeeze(),
                                                                                percentile=95.0)

        if len(hausdorff_scores) == 1:
            self.results.add_value("hausdorff",
                                   next(iter(hausdorff_scores.values())),
                                   name)
            self.results.add_value("hausdorff95",
                                   next(iter(hausdorff95_scores.values())),
                                   name)
        else:
            for (cls, score), (cls_95, score_95) in zip(hausdorff_scores.items(), hausdorff95_scores.items()):
                self.results.add_value("hausdorff_" + cls,
                                       score,
                                       name)
                self.results.add_value("hausdorff95_" + cls_95,
                                       score_95,
                                       name)

            self.results.add_value("hausdorff_mean", hausdorff_mean, name)
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

        displacement = load.load_displacement(path_displacement)
        keypoints_fixed = load.load_keypoints(path_fixed_keypoints)
        keypoints_moving = load.load_keypoints(path_moving_keypoints)

        keypoints_moving_warped = deform_objects.deform_keypoints(keypoints_moving,
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
