from pathlib import Path

from typing import Optional


from registrationbaselines.core import utils, result_csv
from registrationbaselines.transforms import \
    transform_affine_niftyreg, \
    transform_bspline_niftyreg, \
    transform_deformable_corrfield, \
    transform_demons_sitk, \
    transform_syn_ants

# one class for all evaluation metrics
# one .csv file prer registration method
# in yaml it gets the save path (where we save registration reslts)
# in yaml it gets the name of the method and based on that gets the appropriate transformation class


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

        # create the csv file if it doesn't exist
        self.path_results = Path(
            self.configuration['result_path']) / 'results.csv'
        if not self.path_results.exists():
            open(self.path_results, 'w').close()

        self.results = result_csv.EvaluationResults(self.path_results)

    # evaluates all evaluation metrics for a path pair

    def evaluate(self,
                 path_transformation,
                 path_fixed_segmentation: Optional[Path],
                 path_moving_segmentation: Optional[Path],
                 path_fixed_points: Optional[Path],
                 path_moving_points: Optional[Path]) -> None:
        """
        Evaluate the registration model.
        """

        # get the transformaton class based on the method name
        method = self.configuration['method']

        transformation = self.transformation_methods[method](
            self.configuration_path)
