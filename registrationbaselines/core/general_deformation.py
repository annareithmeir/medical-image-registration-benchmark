import SimpleITK as sitk


def apply_displacement_field(image_fixed: sitk.Image,
                             image_moving: sitk.Image,
                             displacement: sitk.Image,
                             sitk_interpolator: int):
    """
    Apply a deformation to an image using the provided deformation.
    """
    # Create the transform using the displacement field
    displacement_field_transform = sitk.DisplacementFieldTransform(
        displacement)

    # Apply the transform to the input image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_fixed)
    resampler.SetInterpolator(sitk_interpolator)
    resampler.SetTransform(displacement_field_transform)

    deformed_image = resampler.Execute(image_moving)

    return deformed_image
