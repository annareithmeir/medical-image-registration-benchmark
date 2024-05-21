from pathlib import Path
import shutil
import warnings
from typing import Optional
from tqdm import tqdm

from registrationbaselines.core import utils, result_csv, utils_metrics
from registrationbaselines.transforms import \
    transform_affine_niftyreg, \
    transform_bspline_niftyreg, \
    transform_deformable_corrfield, \
    transform_demons_sitk, \
    transform_syn_ants
from registrationbaselines.core import metrics

# one .csv file prer registration method


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

        # create the csv file if it doesn't exist
        self.path_results = Path(
            self.configuration['result_path']) / method / 'results.csv'
        if not self.path_results.exists():
            open(self.path_results, 'w').close()

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

    def evaluate(self, dataset_transformations, dataset_data) -> None:
        """
        Evaluate the registration model.
        """

        # TODO this should be removed once we worke with entire datasets
        warnings.warn("Restore the assert, when working with entire datasets.")
        # assert len(dataset_transformations) == len(
        #     dataset_data), "Number of transformations and data must be the same."
        length_datasets = len(dataset_transformations)

        for i in tqdm(range(length_datasets)):
            path_transformation = dataset_transformations[i]
            path_fixed, path_moving = dataset_data.__getitem__(
                i, return_segmentation=True)

            self._evaluate_segmentation(
                path_transformation, path_fixed, path_moving, str(path_fixed.stem).split('.')[0])

            self._evaluate_displacement(
                path_transformation, str(path_fixed.stem).split('.')[0])

        self.results.calculate_mean()
        self.results.calculate_stddev()
        self.results.calculate_min()
        self.results.calculate_max()

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

        dice_scores = {}
        hausdorff_scores = {}

        # check if the segmentation has more than one class. If it has more than one class
        # we have to create new segmentations for each class
        if len(utils_metrics.get_segmentation_classes(path_fixed_segmentation)) == 1:
            # transform the moving segmentation
            self.path_warped = self.transformation.apply_transformation(
                path_fixed_segmentation, path_moving_segmentation, path_transformation)

            # dice coefficient
            dice_scores["dice"] = metrics.dice_score(path_fixed_segmentation,
                                                     self.path_warped)

            # hausdorff distance
            hausdorff_scores["hausdorff"] = metrics.hausdorff_distance(path_fixed_segmentation,
                                                                       self.path_warped)

        else:

            fixed, moving, classes1 = utils_metrics.get_maks_and_classes(
                path_fixed_segmentation, path_moving_segmentation)

            dice_mean = 0
            hausdorff_mean = 0

            # create temp directory
            temp_dir = Path("temp_multi_class_dice")
            temp_dir.mkdir(exist_ok=True)

            for cls in classes1:

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
                dice_scores[f"dice_{int(cls)}"] = current_dice_score
                dice_mean += current_dice_score

                current_hausdorff_score = metrics.hausdorff_distance(
                    path_fixed_temp, self.path_warped)
                hausdorff_scores[f"hausdorff_{int(cls)}"] = current_hausdorff_score
                hausdorff_mean += current_hausdorff_score

            dice_scores["dice_mean"] = dice_mean / len(classes1)
            hausdorff_scores["hausdorff_mean"] = hausdorff_mean / len(classes1)

            # delete all files in the temp directory
            shutil.rmtree(temp_dir)

        for dice_class, dice_value in dice_scores.items():
            self.results.add_value(
                dice_class, dice_value, name)

        for hausdorff_class, hausdorff_value in hausdorff_scores.items():
            self.results.add_value(
                hausdorff_class, hausdorff_value, name)

    def _create_temp_segmentation_file_for_a_class(self, segmentation, path_segmentation, cls, temp_dir):
        class_mask_fixed = utils_metrics.extract_class(
            segmentation.get_fdata(), cls)

        fixed_name = path_segmentation.name.split('.')[0]

        path_fixed_temp = temp_dir / f"{fixed_name}_{cls}.nii.gz"

        utils_metrics.save_class_nifti(segmentation,
                                       class_mask_fixed,
                                       path_fixed_temp)

        return path_fixed_temp
