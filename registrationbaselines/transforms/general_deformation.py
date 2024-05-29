from pathlib import Path

import SimpleITK as sitk


class GeneralDeformation():
    """
    General deformation class used to deform and image with a displacement field using sitk
    """

    def __init__(self, path_results: Path):
        """
        Initialize the transformation.
        """
        if isinstance(path_results, str):
            self.path_results = Path(path_results)
        else:
            self.path_results = path_results

        self.path_deformed = None

    def apply_transformation(self,
                             path_fixed: Path,
                             path_moving: Path,
                             path_deformation: Path,
                             path_output: Path,
                             sitk_interpolator: int):
        """
        Apply a deformation to an image using the provided deformation.
        """
        image_fixed = sitk.ReadImage(path_fixed)
        image_moving = sitk.ReadImage(path_moving)
        displacement = sitk.ReadImage(path_deformation)

        # Ensure the displacement field is of the correct type
        displacement = sitk.Cast(displacement, sitk.sitkVectorFloat64)

        # Create the transform using the displacement field
        displacement_field_transform = sitk.DisplacementFieldTransform(
            displacement)

        # Apply the transform to the input image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_fixed)
        resampler.SetInterpolator(sitk_interpolator)
        resampler.SetTransform(displacement_field_transform)

        deformed_image = resampler.Execute(image_moving)

        path_output = self._save_results(deformed_image, path_output)

        return path_output

    def _save_results(self, deformed, path_output: Path) -> Path:

        sitk.WriteImage(deformed, path_output)

        if not path_output.exists():
            raise ValueError(f"Error saving the file: {path_output}")

        return path_output
