from pathlib import Path

from typing import Tuple

import SimpleITK as sitk
import torch

from registrationbaselines.data_loading import data_loaders
from registrationbaselines.displacement import utils_displacement
from registrationbaselines.interfaces._interface_registration import RegistrationInterface


class DemonsSITK(RegistrationInterface):
    """
    Demosn registration using SimpleITK.
    No default initialisation, as the choice of registration and resampling should be concious.

    # ToDo implement reamining demons
    sitk.DiffeomorphicDemonsRegistrationFilter
    sitk.SymmetricForcesDemonsRegistrationFilter
    sitk.FastSymmetricForcesDemonsRegistrationFilter
    """

    def __init__(self,
                 configuration_path: Path,
                 dataloader: data_loaders.GenericDataset,
                 use_masked_evaluation: bool = True,
                 use_logger=False) -> None:

        super().__init__("DemonsSITK",
                         configuration_path,
                         dataloader,
                         use_masked_evaluation,
                         use_logger=use_logger)

        self.image_fixed: sitk.Image
        self.image_moving: sitk.Image

    def _register(self,
                  fixed_image: torch.Tensor,
                  moving_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a Demons transformation model to register the moving image to the fixed image.
        """

        self.image_fixed = sitk.GetImageFromArray(
            fixed_image.detach().cpu().numpy())
        self.image_moving = sitk.GetImageFromArray(
            moving_image.detach().cpu().numpy())

        # match images
        self._match_images()

        # create displacement field
        result_displacement_field_transform = self._create_displacement_field()

        deformed = self._resample(result_displacement_field_transform)

        # convert transformation to displacement field
        displacement_field = sitk.TransformToDisplacementField(result_displacement_field_transform,
                                                               sitk.sitkVectorFloat64,
                                                               self.image_fixed.GetSize(),
                                                               self.image_fixed.GetOrigin(),
                                                               self.image_fixed.GetSpacing(),
                                                               self.image_fixed.GetDirection())

        displacement = torch.from_numpy(
            sitk.GetArrayFromImage(displacement_field)).to(torch.float32)
        deformed_image = torch.from_numpy(
            sitk.GetArrayFromImage(deformed))

        if not utils_displacement.is_unit_displacement(displacement):
            displacement = utils_displacement.displacement_to_unit_displacement(
                displacement)

        return deformed_image, displacement

    def _match_images(self) -> None:
        matcher = sitk.HistogramMatchingImageFilter()

        if self.image_fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
            matcher.SetNumberOfHistogramLevels(
                self.run_configuration['histogram_levels_int8'])
        else:
            matcher.SetNumberOfHistogramLevels(
                self.run_configuration['histogram_levels_float'])

        matcher.SetNumberOfMatchPoints(
            self.run_configuration['number_of_match_points'])
        matcher.ThresholdAtMeanIntensityOn()

        self.image_moving = matcher.Execute(
            self.image_moving, self.image_fixed)

    def _create_displacement_field(self) -> sitk.DisplacementFieldTransform:

        filter_type = self.run_configuration['filter_type']

        if filter_type == "DemonsRegistrationFilter":
            demons = sitk.DemonsRegistrationFilter()
        elif filter_type == "SymmetricForcesDemonsRegistrationFilter":
            demons = sitk.SymmetricForcesDemonsRegistrationFilter()
        elif filter_type == "FastSymmetricForcesDemonsRegistrationFilter":
            demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        elif filter_type == "DiffeomorphicDemonsRegistrationFilter":
            demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        else:
            raise ValueError("Invalid Demons filter type.")

        demons.SetNumberOfIterations(
            self.run_configuration['number_of_iterations'])

        # Standard deviation for Gaussian smoothing of displacement field
        demons.SetStandardDeviations(
            self.run_configuration['standard_deviations'])

        # get displacement field
        displacement_field = demons.Execute(
            self.image_fixed, self.image_moving)

        return sitk.DisplacementFieldTransform(displacement_field)

    def _resample(self, displacement_field_transform: sitk.DisplacementFieldTransform) -> sitk.Image:
        """
        Resample the moving image using the transformation.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.image_fixed)
        self._set_interpolator(resampler)
        resampler.SetDefaultPixelValue(
            self.run_configuration['default_pixel_value'])
        resampler.SetTransform(displacement_field_transform)

        return resampler.Execute(self.image_moving)

    def _set_interpolator(self, sitk_object: sitk.ResampleImageFilter) -> None:
        if self.run_configuration['interpolator'] == "sitkLinear":
            sitk_object.SetInterpolator(sitk.sitkLinear)
        elif self.run_configuration['interpolator'] == "sitkHammingWindowedSinc":
            sitk_object.SetInterpolator(sitk.sitkHammingWindowedSinc)
        else:
            raise ValueError("Invalid interpolator")
