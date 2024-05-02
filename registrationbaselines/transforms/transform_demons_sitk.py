from pathlib import Path

import SimpleITK as sitk

from registrationbaselines.core import utils_commandline
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
        
        self.fixed_image = None
        self.moving_image = None
    
    def apply_transformation(self,
                             fixed_image_path: Path,
                             moving_image_path: Path,
                             transformation_path: Path):
        """
        Returns the path to the transformed image.
        """
        
        # ensure that the transformation is .h5
        assert transformation_path.suffix == ".tfm", "Transformation should be .tfm"
        
        self.fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        self.moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
        
        self.__match_images()
        
        moving_image_warped = self.__resample(sitk.ReadTransform(transformation_path.as_posix()))
        
        path_output, _ = utils_commandline.create_result_paths(fixed_image_path.parent,
                                                               fixed_image_path.stem,
                                                               moving_image_path.stem,
                                                               self.method,
                                                               ".nii",
                                                               ".tfm")

        sitk.WriteImage(moving_image_warped, path_output.as_posix())
        
        if not path_output.exists():
            raise ValueError(f"Error saving the file: {path_output}")
        
        return path_output
    
    def __match_images(self) -> None:
        matcher = sitk.HistogramMatchingImageFilter()
        
        if self.fixed_image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
            matcher.SetNumberOfHistogramLevels(self.config_resample['histogram_levels_int8'])
        else:
            matcher.SetNumberOfHistogramLevels(self.config_resample['histogram_levels_float'])
            
        matcher.SetNumberOfMatchPoints(self.config_resample['number_of_match_points'])
        matcher.ThresholdAtMeanIntensityOn()
        
        self.moving_image = matcher.Execute(self.moving_image, self.fixed_image)
    
    def __resample(self, transformation):
        """
        Resample the moving image using the transformation.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.fixed_image)
        self.__set_interpolator(resampler)
        resampler.SetDefaultPixelValue(self.config_resample['default_pixel_value'])
        resampler.SetTransform(transformation)

        return resampler.Execute(self.moving_image)
    
    def __set_interpolator(self, sitk_object):
        if self.config_resample['interpolator'] == "sitkLinear":
            sitk_object.SetInterpolator(sitk.sitkLinear)
        elif self.config_resample['interpolator'] == "sitkHammingWindowedSinc":
            sitk_object.SetInterpolator(sitk.sitkHammingWindowedSinc)
        else:
            raise ValueError("Invalid interpolator")