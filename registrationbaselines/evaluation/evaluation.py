from pathlib import Path

from typing import Optional, Tuple

import wandb
import numpy as np
import torch

from registrationbaselines.evaluation import result_csv, plot_objects, utils_evaluation
from registrationbaselines.io import load, save
from registrationbaselines.displacement import deform_objects
from registrationbaselines.metrics import metrics


class RegistrationEvaluator():
    """
    Class for evaluation of registration methods.

    It requires a precomputed transformation.
    """

    def __init__(self,
                 run_path: Path,
                 number_of_images: int) -> None:
        """
        Initialize the evaluatin model.
        """

        # create the csv file and all its parents if doesn't exist
        self.path_results = run_path / 'results.csv'

        self.path_plots = run_path / 'plots'
        self.path_results.parent.mkdir(parents=True, exist_ok=True)
        self.path_plots.mkdir(parents=True, exist_ok=True)
        self.path_results.touch()

        self.results = result_csv.EvaluationMetricsResults(self.path_results,
                                                           number_of_images)

    def evaluate(self,
                 row_name: str,
                 displacement: Optional[torch.Tensor] = None,
                 fixed_segmentations: Optional[Tuple[torch.Tensor, str]] = None,
                 moving_segmentations: Optional[Tuple[torch.Tensor, str]] = None,
                 fixed_evaluation_mask: Optional[torch.Tensor] = None
                 ) -> None:
        """
        Evaluate the registration model.
        """

        if displacement is not None:
            self._evaluate_displacement(displacement,
                                        row_name,
                                        fixed_evaluation_mask)

        if fixed_segmentations[0] is not None and moving_segmentations[0] is not None:
            self._evaluate_segmentation(fixed_segmentations[0],
                                        fixed_segmentations[1],
                                        moving_segmentations[0],
                                        moving_segmentations[1],
                                        row_name,
                                        displacement,
                                        fixed_evaluation_mask)

    def visualize(self,
                  fixed_image: torch.Tensor,
                  fixed_image_name: str,
                  moving_image: torch.Tensor,
                  moving_image_name: str,
                  warped_image: Optional[torch.Tensor] = None,
                  displacement: Optional[torch.Tensor] = None,
                  fixed_segmentations: Optional[torch.Tensor] = None,
                  moving_segmentations: Optional[torch.Tensor] = None,
                  fixed_evaluation_mask: Optional[torch.Tensor] = None
                  ) -> None:
        """
        Create plots for the evaluation.
        """

        if displacement is None and moving_segmentations is not None:
            device = fixed_image.device

            deformed_segmentations = moving_segmentations.detach().clone()
            shape = fixed_image.shape
            displacement = torch.zeros(
                (*shape, 3), dtype=torch.float32).to(device)

        elif fixed_segmentations is None or moving_segmentations is None:
            deformed_segmentations = None
            device = displacement.device

        else:
            deformed_segmentations = deform_objects.deform_image(moving_segmentations,
                                                                 displacement)
            device = displacement.device

            fixed_segmentations = fixed_segmentations.to(device)
            moving_segmentations = moving_segmentations.to(device)

        if fixed_evaluation_mask is not None and deformed_segmentations is not None and moving_segmentations is not None:
            deformed_segmentations *= fixed_evaluation_mask
            moving_segmentations *= fixed_evaluation_mask

        plots_path = self._create_plots_paths(fixed_image_name,
                                              moving_image_name,
                                              extension_overwrite='.nii.gz')

        fixed_image = fixed_image.to(device)
        moving_image = moving_image.to(device)

        plot_objects.plot_all_registration_results(moving_image=moving_image,
                                                   fixed_image=fixed_image,
                                                   displacement=displacement,
                                                   pred_image=warped_image,
                                                   fixed_segmentations=fixed_segmentations,
                                                   pred_segmentations=deformed_segmentations,
                                                   save_path=plots_path)

    def _evaluate_displacement(self,
                               displacement: torch.Tensor,
                               row_name: str,
                               fixed_evaluation_mask: Optional[torch.Tensor] = None) -> None:
        """
        Evaluates the displacement field with sdlogj and fraction of foldings.

        @param path_displacement: Path to the displacement field (torch tensor or nifti file).
        @param row_name: row_name of the evaluated file pair
        """

        sd_log_det, fraction_foldings = metrics.displacement_field_metrics(
            displacement,
            None)  # fixed_evaluation_mask)

        self.results.add_value("sdlogj", sd_log_det, row_name=row_name)
        self.results.add_value("frac_foldings", fraction_foldings, row_name)

    def _evaluate_segmentation(self,
                               segmentations_fixed: torch.Tensor,
                               segmentations_fixed_name: str,
                               segmentations_moving: torch.Tensor,
                               segmentations_moving_name: str,
                               row_name: str,
                               displacement: Optional[torch.Tensor] = None,
                               fixed_evaluation_mask: Optional[torch.Tensor] = None) -> None:
        """
        Evaluate segmentations.

        If there is only one class it is trivial. If there are more classes, we have to
        create new temporary segmentation files for each class and evaluate them
        (because of interpolation issues).

        Args:

            path_displacement (Path): The path to the transformation file.
            path_segmentation_fixed (Path): The path to the fixed segmentation file.
            path_segmentation_moving (Path): The path to the moving segmentation file.
            row_name (str): The row_name (name of the image pair).

        Returns:
            None
        """

        if displacement is not None:
            segmentation_warped = deform_objects.deform_image(segmentations_moving,
                                                              displacement)
            # save the deformed segmentation
            deformed_segmentation_path = self._get_deformed_image_path(segmentations_fixed_name,
                                                                       segmentations_moving_name,
                                                                       extension_overwrite='.nii.gz')
            deformed_segmentation_path = Path(
                deformed_segmentation_path.as_posix().replace(".nii", "_seg.nii"))
            save.save_segmentation(segmentation_warped,
                                   deformed_segmentation_path)

        else:
            shape = segmentations_fixed.shape
            displacement = torch.zeros((*shape, 3), dtype=torch.float32)
            segmentation_warped = segmentations_moving.detach().clone()

        if fixed_evaluation_mask is not None:
            segmentation_warped *= fixed_evaluation_mask

        dice_scores, dice_mean = metrics.dice_score(segmentations_fixed,
                                                    segmentation_warped)

        if len(dice_scores) == 1:
            self.results.add_value("dice",
                                   next(iter(dice_scores.values())),
                                   row_name)
        else:
            for cls, score in dice_scores.items():
                self.results.add_value("dice_" + cls,
                                       score,
                                       row_name)

            self.results.add_value("dice_mean",
                                   dice_mean,
                                   row_name)

        hausdorff_scores, hausdorff_mean = metrics.hausdorff_distance_monai(segmentations_fixed.squeeze(),
                                                                            segmentation_warped.squeeze())
        hausdorff95_scores, hausdorff95_mean = metrics.hausdorff_distance_monai(segmentations_fixed.squeeze(),
                                                                                segmentation_warped.squeeze(),
                                                                                percentile=95.0)

        if len(hausdorff_scores) == 1:
            self.results.add_value("hausdorff",
                                   next(iter(hausdorff_scores.values())),
                                   row_name)
            self.results.add_value("hausdorff95",
                                   next(iter(hausdorff95_scores.values())),
                                   row_name)
        else:
            for (cls, score), (cls_95, score_95) in zip(hausdorff_scores.items(), hausdorff95_scores.items()):
                self.results.add_value("hausdorff_" + cls,
                                       score,
                                       row_name)
                self.results.add_value("hausdorff95_" + cls_95,
                                       score_95,
                                       row_name)

            self.results.add_value("hausdorff_mean", hausdorff_mean, row_name)
            self.results.add_value(
                "hausdorff95_mean", hausdorff95_mean, row_name)

    def _evaluate_keypoints(self,
                            path_displacement: Path,
                            path_fixed_keypoints: Path,
                            path_moving_keypoints: Path,
                            row_name: str) -> None:

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

        self.results.add_value("tre", tre, row_name)
        self.results.add_value("tre30", tre30, row_name)

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
