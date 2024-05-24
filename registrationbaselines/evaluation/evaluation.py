from pathlib import Path
import shutil
import warnings
from typing import Optional
from tqdm import tqdm
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

from registrationbaselines.core import utils, result_csv, utils_metrics
from registrationbaselines.transforms import \
    transform_affine_niftyreg, \
    transform_bspline_niftyreg, \
    transform_deformable_corrfield, \
    transform_demons_sitk, \
    transform_syn_ants
from registrationbaselines.core import metrics
from registrationbaselines.core import visualization


# one .csv file per registration method


class Evaluation():
    """
    Class for evaluation of registration methods.

    It requires a precomputed transformation.
    """

    # create a dictionary with method names and the transformation classes
    transformation_methods = {
        "AffineNiftyReg": transform_affine_niftyreg.TransformAffineNiftyReg,
        "BSplineNiftyReg": transform_bspline_niftyreg.TransformBSplineNiftyReg,
        "DeformableCorrField": transform_deformable_corrfield.TransformDeformableCorrField,
        "DemonsSITK": transform_demons_sitk.TransformDemonsSITK,
        "SyNANTs": transform_syn_ants.TransformSyNANTs
    }

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
        self.path_plots = Path(
            self.configuration['result_path']) / method / 'plots'
        self.path_results.parent.mkdir(parents=True, exist_ok=True)
        self.path_plots.mkdir(parents=True, exist_ok=True)
        self.path_results.touch()

        self.results = result_csv.EvaluationResults(self.path_results)

        # get the transformaton class based on the method name
        self.transformation = self.transformation_methods[method](
            self.configuration_path)

        self.transformation.path_deformed = self.transformation.path_deformed / "temp"

        self.path_warped = None

    def __del__(self):
        """
        Clean up the temporary directory (called at the end of scope to delete the temporary directory).
        """

        if self.transformation.path_deformed is not None and self.transformation.path_deformed.exists():
            shutil.rmtree(self.transformation.path_deformed)

    # TODO first column should contain names of the fixed image only

    def evaluate(self, dataset_transformations: Dataset, dataset_data: Dataset) -> None:
        """
        Evaluate the registration model.
        """

        # TODO this should be removed once we worke with entire datasets
        warnings.warn("Restore the assert, when working with entire datasets.")
        # assert len(dataset_transformations) == len(
        #     dataset_data), "Number of transformations and data must be the same."
        length_datasets = len(dataset_transformations)

        self.dataset_data = dataset_data

        for i in tqdm(range(length_datasets)):
            path_transformation = dataset_transformations[i]
            item = dataset_data[i]

            self._evaluate_displacement(
                path_transformation, str(item["imgs"][0].stem).split('.')[0])

            if "segs" in item:
                path_moving = item["segs"][0]
                path_fixed = item["segs"][1]

                self._evaluate_segmentation(
                    path_transformation, path_fixed, path_moving, str(path_fixed.stem).split('.')[0])

            if "kps" in item:
                path_moving_landmarks = item["kps"][0]
                path_fixed_landmarks = item["kps"][1]
                self._evaluate_landmarks(
                    path_transformation, path_fixed_landmarks, path_moving_landmarks,
                    str(path_fixed.stem).split('.')[0])

        self.results.calculate_mean()
        self.results.calculate_stddev()
        self.results.calculate_min()
        self.results.calculate_max()

        self.results.write()

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
            moving_image_path = item["imgs"][0]
            fixed_image_path = item["imgs"][1]
            moving_image = nib.load(moving_image_path).get_fdata()
            fixed_image = nib.load(fixed_image_path).get_fdata()
            displacement = nib.load(path_transformation.as_posix()).get_fdata().squeeze()
            deformed_image_path = self._get_deformed_image_path(moving_image_path.name, fixed_image_path.name)
            deformed_image = nib.load(deformed_image_path).get_fdata()

            plots_path = self._create_plots_paths(fixed_image_path.name, moving_image_path.name)

            if "segs" in item:
                moving_segmentation = nib.load(item["segs"][0]).get_fdata()
                fixed_segmentation = nib.load(item["segs"][1]).get_fdata()
                #deformed_segmentation = utils_metrics.deform_segmentations(moving_segmentation, displacement)
                deformed_segmentation = None # TODO implement function above
            if "kps" in item:
                moving_landmarks = np.genfromtxt(item["kps"][0], delimiter=',')
                fixed_landmarks = np.genfromtxt(item["kps"][1], delimiter=',')
                deformed_landmarks = utils_metrics.deform_landmarks(moving_landmarks, displacement)
            visualization.plot_all_registration_results(plots_path, moving_image, fixed_image, deformed_image,
                                                        displacement, fixed_segmentation, deformed_segmentation,
                                                        fixed_landmarks, moving_landmarks, deformed_landmarks)

    def _evaluate_displacement(self, path_displacement: Path, name: str) -> None:

        sd_log_det, num_foldings = metrics.sdlogj(path_displacement)

        self.results.add_value("sdlogj", sd_log_det, name)
        self.results.add_value("num_foldings", num_foldings, name)

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

        fixed, moving, classes1 = utils_metrics.get_maks_and_classes(
            path_fixed_segmentation, path_moving_segmentation)

        dice_mean = 0
        hausdorff_mean = 0
        hausdorff95_mean = 0

        # create temp directory
        temp_dir = Path("temp_multi_class_dice")
        temp_dir.mkdir(exist_ok=True)

        for cls in classes1:

            if len(classes1) == 1:
                postfix = ""
            else:
                postfix = f"_{int(cls)}"

            path_fixed_temp = self._create_temp_segmentation_file_for_a_class(fixed,
                                                                              path_fixed_segmentation,
                                                                              cls,
                                                                              temp_dir)
            path_moving_temp = self._create_temp_segmentation_file_for_a_class(moving,
                                                                               path_moving_segmentation,
                                                                               cls,
                                                                               temp_dir)

            self.path_warped = self.transformation.apply_transformation(
                path_fixed_temp, path_moving_temp, path_transformation)

            current_dice_score = metrics.dice_score(
                path_fixed_temp, self.path_warped)
            self.results.add_value(
                f"dice" + postfix, current_dice_score, name)
            dice_mean += current_dice_score

            current_hausdorff_score = metrics.hausdorff_distance(
                path_fixed_temp, self.path_warped)
            self.results.add_value(
                f"hausdorff" + postfix, current_hausdorff_score, name)
            hausdorff_mean += current_hausdorff_score

            current_hausdorff95_score = metrics.hausdorff_distance(
                path_fixed_temp, self.path_warped, percentile=95)
            self.results.add_value(
                f"hausdorff95" + postfix, current_hausdorff95_score, name)
            hausdorff95_mean += current_hausdorff95_score

        if len(classes1) > 1:
            self.results.add_value(
                "dice_mean", dice_mean / len(classes1), name)
            self.results.add_value(
                "hausdorff_mean", hausdorff_mean / len(classes1), name)
            self.results.add_value(
                "hausdorff95_mean", hausdorff95_mean / len(classes1), name)

        # delete all files in the temp directory
        shutil.rmtree(temp_dir)

    def _evaluate_landmarks(self,
                            path_transformation: Path,
                            path_fixed_landmarks: Path,
                            path_moving_landmarks: Path,
                            name: str) -> None:

        assert self.dataset_data is not None

        tre = metrics.tre(path_fixed_landmarks, path_moving_landmarks, path_transformation, self.dataset_data.spacing)
        tre30 = metrics.tre(path_fixed_landmarks, path_moving_landmarks, path_transformation, self.dataset_data.spacing,
                            percentile=30)

        self.results.add_value("tre", tre, name)
        self.results.add_value("tre30", tre30, name)

    def _create_temp_segmentation_file_for_a_class(self, segmentation, path_segmentation, cls, temp_dir):
        class_mask_fixed = utils_metrics.extract_class(
            segmentation.get_fdata(), cls)

        fixed_name = path_segmentation.name.split('.')[0]

        path_fixed_temp = temp_dir / f"{fixed_name}_{cls}.nii.gz"

        utils_metrics.save_class_nifti(segmentation,
                                       class_mask_fixed,
                                       path_fixed_temp)

        return path_fixed_temp

    def _create_plots_paths(self, name_fixed: str, name_moving: str):
        """
        Create the paths for the plots.
        """

        name_moving = name_moving.replace(".nii", "")
        name_fixed = name_fixed.replace(".nii", "")

        name_moving = name_moving.replace(".gz", "")
        name_fixed = name_fixed.replace(".gz", "")

        path_plots = self.path_plots / f"{name_moving}_deformed_to_{name_fixed}.pdf"
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

        path_plots = self.path_results.parent / f"deformed/{name_moving}_deformed_to_{name_fixed}.nii"
        path_plots = path_plots.resolve().as_posix()

        return Path(path_plots)
