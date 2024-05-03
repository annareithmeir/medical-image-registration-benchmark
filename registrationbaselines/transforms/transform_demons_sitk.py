from pathlib import Path

import SimpleITK as sitk

from registrationbaselines.transforms._interface_transformation import TransformationInterface


class TransformDemonsSITK(TransformationInterface):
    """
    SyN transformation using ANTs.
    """

    def __init__(self, configuration_path_resample: Path) -> None:
        """
        Initialize the transformation model.
        """

        self.method = "DemonsSITK"

        self.config_resample = self.read_config(configuration_path_resample)

        self.__create_result_directories()

        self.fixed_image = None
        self.moving_image = None

        self.path_fixed = Path()
        self.path_moving = Path()

    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """

        # ensure that the transformation is .h5
        assert transformation_path.suffix == ".tfm", "Transformation should be .tfm"

        self.path_fixed = fixed_image_path
        self.path_moving = moving_image_path

        self.fixed_image = sitk.ReadImage(self.path_fixed, sitk.sitkFloat32)
        self.moving_image = sitk.ReadImage(self.path_moving, sitk.sitkFloat32)

        self.__match_images()

        moving_image_warped = self.__resample(
            sitk.ReadTransform(transformation_path.as_posix()))

        path_output = self.__save_results(moving_image_warped)

        return path_output

    def __save_results(self, deformed):
        path_deformed = self.__create_result_path(self.path_fixed.stem,
                                                  self.path_moving.stem,
                                                  ".nii")

        sitk.WriteImage(deformed, path_deformed)

        if not path_deformed.exists():
            raise ValueError(f"Error saving the file: {path_deformed}")

        return path_deformed

    def __match_images(self) -> None:
        matcher = sitk.HistogramMatchingImageFilter()

        if self.fixed_image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
            matcher.SetNumberOfHistogramLevels(
                self.config_resample['histogram_levels_int8'])
        else:
            matcher.SetNumberOfHistogramLevels(
                self.config_resample['histogram_levels_float'])

        matcher.SetNumberOfMatchPoints(
            self.config_resample['number_of_match_points'])
        matcher.ThresholdAtMeanIntensityOn()

        self.moving_image = matcher.Execute(
            self.moving_image, self.fixed_image)

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
