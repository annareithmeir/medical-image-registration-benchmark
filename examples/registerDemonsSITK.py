import SimpleITK as sitk
import os


from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute().parent))

from registrationbaselines.registration.affine_sitk import AffineSITK


def main() -> None:

    registration = AffineSITK(Path('registrationbaselines/configs/AffineSITK.yaml'), Path('registrationbaselines/configs/ResampleSITK.yaml'))

    registration.register(Path("registrationbaselines/data/unregistered/tumor1.nii"), Path("registrationbaselines/data/unregistered/tumor2.nii"))


def original():
        
    def command_iteration(filter):
        print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")


    fixed = sitk.ReadImage("/home/fryderyk/Documents/code/registrationbaselines/registrationbaselines/data/affinely_registered_NiftyReg/tumor1_resampled111_normalized.nii.gz")
    moving = sitk.ReadImage("/home/fryderyk/Documents/code/registrationbaselines/registrationbaselines/data/affinely_registered_NiftyReg/tumor2_resampled111_normalized.nii.gz")

    # TODO images need to be of the same size

    matcher = sitk.HistogramMatchingImageFilter()
    if fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    # The fast symmetric forces Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in
    # SimpleITK
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(200)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(1.0)

    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

    displacementField = demons.Execute(fixed, moving)

    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f" RMS: {demons.GetRMSChange()}")

    outTx = sitk.DisplacementFieldTransform(displacementField)

    sitk.WriteTransform(outTx, "/home/fryderyk/Documents/code/registrationbaselines/registrationbaselines/data/transform.tfm")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    sitk.WriteImage(out, "/home/fryderyk/Documents/code/registrationbaselines/registrationbaselines/data/registered.nii.gz")

if __name__ == "__main__":
    # main()
    
    original()
