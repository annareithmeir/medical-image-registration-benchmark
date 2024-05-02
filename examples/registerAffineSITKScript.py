"""
Script to affinely register two images.
"""

import SimpleITK as sitk


def main():

    fixed = sitk.ReadImage(r"registrationbaselines/data/unregistered/tumor1.nii", sitk.sitkFloat32)

    moving = sitk.ReadImage(r"registrationbaselines/data/unregistered/tumor2.nii", sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsCorrelation()

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()

    # affine initial transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    R.SetInitialTransform(initial_transform)

    R.SetInterpolator(sitk.sitkLinear)

    outTx = R.Execute(fixed, moving)

    sitk.WriteTransform(outTx, r"transform.hdf5")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    sitk.WriteImage(out, r"output.nii")


if __name__ == "__main__":
    main()
