from pathlib import Path

import SimpleITK as sitk

from registrationbaselines.registration._interface_registration import RegistrationInterface


class DemonsSITK(RegistrationInterface):
    """
    Demosn registration using SimpleITK.
    No default initialisation, as the choice of registration and resampling should be concious.
    """

    def __init__(self, configuration_path: Path) -> None:

        self.method = "DemonsSITK"

        # configuration
        self.configuration = self.read_config(configuration_path)

        self._create_result_directories()

        # paths
        self.fixed_path: Path
        self.moving_path: Path
        self.result_transformed_image_path: Path
        self.result_transformation_path: Path
        self.working_dir_path: Path

        self.fixed_image: sitk.Image
        self.moving_image: sitk.Image

    def register(self, fixed_image_path: Path,
                 moving_image_path: Path,
                 print_progress: bool = False):
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

        # convert transformation to displacement field
        displacement_field = sitk.TransformToDisplacementField(result_transformation,
                                                               sitk.sitkVectorFloat64,
                                                               self.fixed_image.GetSize(),
                                                               self.fixed_image.GetOrigin(),
                                                               self.fixed_image.GetSpacing(),
                                                               self.fixed_image.GetDirection())

        self._save_results(result_transformed_image, displacement_field)

    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation

        return self.result_transformation_path

    def _save_results(self, deformed, deformation):
        self.result_transformed_image_path, \
            self.result_transformation_path = self._create_result_paths(self.fixed_path.stem,
                                                                        self.moving_path.stem,
                                                                        ".nii.gz",
                                                                        ".nii.gz")

        sitk.WriteImage(deformed, self.result_transformed_image_path)
        sitk.WriteImage(deformation, self.result_transformation_path)

    def __match_images(self) -> None:
        matcher = sitk.HistogramMatchingImageFilter()

        if self.fixed_image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
            matcher.SetNumberOfHistogramLevels(
                self.configuration['histogram_levels_int8'])
        else:
            matcher.SetNumberOfHistogramLevels(
                self.configuration['histogram_levels_float'])

        matcher.SetNumberOfMatchPoints(
            self.configuration['number_of_match_points'])
        matcher.ThresholdAtMeanIntensityOn()

        self.moving_image = matcher.Execute(
            self.moving_image, self.fixed_image)

    def __create_displacement_field(self) -> sitk.DisplacementFieldTransform:

        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(
            self.configuration['number_of_iterations'])

        # Standard deviation for Gaussian smoothing of displacement field
        demons.SetStandardDeviations(
            self.configuration['standard_deviations'])

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
            self.configuration['default_pixel_value'])
        resampler.SetTransform(transformation)

        return resampler.Execute(self.moving_image)

    def __set_interpolator(self, sitk_object):
        if self.configuration['interpolator'] == "sitkLinear":
            sitk_object.SetInterpolator(sitk.sitkLinear)
        elif self.configuration['interpolator'] == "sitkHammingWindowedSinc":
            sitk_object.SetInterpolator(sitk.sitkHammingWindowedSinc)
        else:
            raise ValueError("Invalid interpolator")
