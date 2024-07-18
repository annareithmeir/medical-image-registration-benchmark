from pathlib import Path
import warnings

from typing import Optional, Tuple, Any

from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import torch
import wandb

from registrationbaselines.core import utils, result_csv
from registrationbaselines.core import metrics
from registrationbaselines.core import visualization
from registrationbaselines.data_loading.data_loaders import BaselineTransformations
from registrationbaselines.core.types import floatArray3Dor4D, floatArray2Dor3D


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

    def evaluate(self, dataset_transformations: BaselineTransformations, dataset_data: Dataset[Any]) -> None:
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

            fixed_name = str(item["fixed_image"].stem).split('.')[0]

            self._evaluate_displacement(path_displacement, fixed_name)

            path_fixed = item["fixed_labels"]
            path_moving = item["moving_labels"]

            self._evaluate_segmentation(path_displacement,
                                        path_fixed,
                                        path_moving,
                                        fixed_name)

            """
            if "landmarks" in item:
                # is2d = sitk.GetArrayFromImage(utils.load_image(
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
            """

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

        for i in tqdm(idxs):
            path_displacement = dataset_transformations[i]
            item = dataset_data[i]
            fixed_image_path = item["img_x"][0]
            moving_image_path = item["img_y"][0]
            fixed_image = torch.from_numpy(item["img_x"][1])
            moving_image = torch.from_numpy(item["img_y"][1])
            displacement = torch.load(path_displacement)

            deformed_image_path = self._get_deformed_image_path(fixed_image_path.name,
                                                                moving_image_path.name,
                                                                extension_overwrite=''.join(path_displacement.suffixes))
            deformed_image = torch.load(deformed_image_path)

            plots_path = self._create_plots_paths(fixed_image_path.name,
                                                  moving_image_path.name,
                                                  extension_overwrite=''.join(path_displacement.suffixes))

            fixed_landmarks = None
            moving_landmarks = None
            deformed_landmarks = None
            fixed_segmentation = None
            deformed_segmentation = None

            fixed_segmentation = torch.from_numpy(
                item["labels_x"][1]).to(displacement.device)
            moving_segmentation = torch.from_numpy(
                item["labels_y"][1]).to(displacement.device)

            deformed_segmentation = utils.deform_image(moving_segmentation,
                                                       displacement, mode='nearest')

            fixed_image = fixed_image.to(displacement.device)
            moving_image = moving_image.to(displacement.device)

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
            visualization.plot_all_registration_results(plots_path, moving_image, fixed_image, deformed_image,
                                                        displacement, fixed_labels=fixed_segmentation, pred_labels=deformed_segmentation,
                                                        fixed_keypoints=None, moving_keypoints=None, pred_keypoints=None)

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
        if not suffixes == [".pt"] and \
            not suffixes == [".nii"] and \
                not suffixes == [".nii", ".gz"]:
            raise ValueError(
                f"Displacement file should have suffixes '.pt', '.nii' or '.nii.gz' but has {suffixes}.")

        if suffixes == [".pt"]:
            displacement = torch.load(
                path_displacement).detach().cpu().numpy().squeeze()
        else:
            displacement = utils.load_image(path_displacement).squeeze()

        sd_log_det, fraction_foldings = metrics.displacement_field_metrics(
            displacement)

        self.results.add_value("sdlogj", sd_log_det, name)
        self.results.add_value("frac_foldings", fraction_foldings, name)

    def dice_anna(self, image1: np.ndarray, image2: np.ndarray, img_mask: Optional[np.ndarray] = None) -> float:
        """
        Taken from github of conditional LapIRN by Tony Mok
        :param image1: pred
        :param image2:true
        :return:
        """
        unique_class = np.unique(image2)
        dice = 0
        num_count = 0
        if img_mask is not None:
            image1[img_mask == 0] = 0
            # image2[img_mask==0]=0
        for i in unique_class:
            if (i == 0) or ((image1 == i).sum() == 0) or ((image2 == i).sum() == 0):
                continue
            sub_dice = np.sum(image2[image1 == i] == i) * \
                2.0 / (np.sum(image1 == i) + np.sum(image2 == i))
            dice += sub_dice
            num_count += 1
        if num_count == 0:
            return 0
        else:
            return dice / num_count

    def dice_per_class_anna(self, image1: np.ndarray, image2: np.ndarray, classes: list[int], img_mask: np.ndarray = None) -> list[float]:
        """
        Computes dice scores per class labels. Based on Tony Mok LapIRN implementation
        :param image1:
        :param image2:
        :param classes: list of labels to compute dice on
        :return: list of dice scores
        """
        dice_ls = []
        if img_mask is not None:
            image1[img_mask == 0] = 0
        for i in classes:
            if (i == 0) or (np.sum(image1 == i) == 0) or (np.sum(image2 == i) == 0):
                dice_ls.append(0)  # TODO check if correct
                continue
            sub_dice = np.sum(image2[image1 == i] == i) * \
                2.0 / (np.sum(image1 == i) + np.sum(image2 == i))
            dice_ls.append(sub_dice)
        return dice_ls

    def _evaluate_segmentation(self,
                               path_displacement: Path,
                               segmentation_fixed: Tuple[Path, np.ndarray],
                               segmentation_moving: Tuple[Path, np.ndarray],
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

        if not path_displacement.exists():
            raise FileNotFoundError(
                f"File {path_displacement} does not exist.")

        suffixes = path_displacement.suffixes
        if not suffixes == [".pt"] and \
            not suffixes == [".nii"] and \
                not suffixes == [".nii", ".gz"]:
            raise ValueError(
                f"Displacement file should have suffixes '.pt', '.nii' or '.nii.gz' but has {suffixes}.")

        if suffixes == [".pt"]:
            displacement = torch.load(
                path_displacement).detach().cpu().numpy().squeeze()
        else:
            displacement = utils.load_image(path_displacement).squeeze()

        displacement = torch.load(path_displacement)
        segmentation_fixed = torch.from_numpy(
            segmentation_fixed[1]).to(displacement.device)
        segmentation_moving = torch.from_numpy(
            segmentation_moving[1]).to(displacement.device)

        warped = utils.deform_image(segmentation_moving,
                                    displacement, mode='nearest')

        dice_scores = metrics.dice_score(
            segmentation_fixed.squeeze(), warped.squeeze())

        # dice_score_me = np.mean(np.array(dice_scores))
        # dice_scores_anna = self.dice_per_class_anna(segmentation_fixed.squeeze(
        # ).detach().cpu().numpy(), segmentation_moving.squeeze().detach().cpu().numpy(), [1, 2, 3])

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

    def _get_deformed_image_path(self, name_fixed: str, name_moving: str, extension_overwrite=None):
        """
        Get the corresponding deformed image path.
        """

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

        path_plots = self.path_results.parent / \
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
