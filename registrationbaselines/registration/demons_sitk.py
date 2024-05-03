from pathlib import Path

import SimpleITK as sitk

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_commandline


class DemonsSITK(RegistrationInterface):
    """
    Demosn registration using SimpleITK.
    No default initialisation, as the choice of registration and resampling should be concious.
    """

    def __init__(self,
                 configuration_path_registration: Path,
                 configuration_path_resample: Path) -> None:
        # todo make method an enum
        self.method = "DemonsSITK"

        # registration configuration
        self.config_reg = self.read_config(configuration_path_registration)

        # resample configuration
        self.config_resample = self.read_config(configuration_path_resample)

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.result_transformation_path = Path()
        self.working_dir_path = Path()

        self.fixed_image = None
        self.moving_image = None

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):
        """
        Creates a Demons transformation model to register the moving image to the fixed image.


        """

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path
        self.working_dir_path = self.fixed_path.parent

        # check that both images exist
        assert self.fixed_path.exists(
        ), f"File {self.fixed_path} does not exist."
        assert self.moving_path.exists(
        ), f"File {self.moving_path} does not exist."

        self.fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        self.moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

        # match images
        self.__match_images()

        # create displacement field
        result_transformation = self.__create_displacement_field()

        result_transformed_image = self.__resample(result_transformation)

        self.__save_results(result_transformed_image, result_transformation)

    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation

        return self.result_transformation_path

    def __save_results(self, deformed, deformation):
        self.result_transformed_image_path, self.result_transformation_path = utils_commandline.create_result_paths(self.working_dir_path,
                                                                                                                    self.fixed_path.stem,
                                                                                                                    self.moving_path.stem,
                                                                                                                    self.method,
                                                                                                                    ".nii",
                                                                                                                    ".tfm")

        sitk.WriteImage(deformed, self.result_transformed_image_path)
        sitk.WriteTransform(deformation, self.result_transformation_path)

    def __match_images(self) -> None:
        matcher = sitk.HistogramMatchingImageFilter()

        if self.fixed_image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
            matcher.SetNumberOfHistogramLevels(
                self.config_reg['histogram_levels_int8'])
        else:
            matcher.SetNumberOfHistogramLevels(
                self.config_reg['histogram_levels_float'])

        matcher.SetNumberOfMatchPoints(
            self.config_reg['number_of_match_points'])
        matcher.ThresholdAtMeanIntensityOn()

        self.moving_image = matcher.Execute(
            self.moving_image, self.fixed_image)

    def __create_displacement_field(self) -> sitk.DisplacementFieldTransform:

        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(self.config_reg['number_of_iterations'])

        # Standard deviation for Gaussian smoothing of displacement field
        demons.SetStandardDeviations(self.config_reg['standard_deviations'])

        # get displacement field
        displacement_field = demons.Execute(
            self.fixed_image, self.moving_image)

        return sitk.DisplacementFieldTransform(displacement_field)

    def __resample(self, transformation):
        """
        Resample the moving image using the transformation.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.fixed_image)
        self.__set_interpolator(resampler)
        resampler.SetDefaultPixelValue(
            self.config_resample['default_pixel_value'])
        resampler.SetTransform(transformation)

        return resampler.Execute(self.moving_image)

    def __set_interpolator(self, sitk_object):
        if self.config_resample['interpolator'] == "sitkLinear":
            sitk_object.SetInterpolator(sitk.sitkLinear)
        elif self.config_resample['interpolator'] == "sitkHammingWindowedSinc":
            sitk_object.SetInterpolator(sitk.sitkHammingWindowedSinc)
        else:
            raise ValueError("Invalid interpolator")
