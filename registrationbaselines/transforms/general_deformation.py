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
        displacement_file = sitk.ReadImage(
            path_deformation, sitk.sitkVectorFloat64)

        # get the data from the displacement field
        displacement = sitk.GetArrayFromImage(displacement_file).squeeze()

        assert displacement.ndim == 4, "Displacement field should have shape (h, w, d, 3) or (h, w, d, 3)"

        if displacement.shape[-1] != 3 and displacement.shape[0] == 3:
            displacement = displacement.transpose(1, 2, 3, 0)

        # Create the displacement field image again
        displacement = sitk.GetImageFromArray(displacement, isVector=True)

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
