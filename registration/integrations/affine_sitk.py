from pathlib import Path
import datetime

import SimpleITK as sitk

from ..core.enums import SITKSimilarityMetric, SITKOptimizer
from ..core.registration_interface import RegistrationInterface
from ..core.configurations import AffineSITKConfiguration, ResampleSITKConfiguration


class AffineSITK(RegistrationInterface):
    """
    Affine registration using SimpleITK.
    No default initialisation, as the choice of registration and resampling should be concious.
    """

    def __init__(self,
                 configuration_registration: AffineSITKConfiguration,
                 configuration_resample: ResampleSITKConfiguration):

        self.method = "AffneSITK"

        # registration configuration
        self.similarity_metric = configuration_registration.similarity_metric
        self.optimizer = configuration_registration.optimizer
        self.learning_rate = configuration_registration.learning_rate
        self.min_step = configuration_registration.min_step
        self.number_of_iterations = configuration_registration.number_of_iterations
        self.gradient_magnitude_tolerance = configuration_registration.gradient_magnitude_tolerance
        self.interpolator_registration = configuration_registration.interpolator

        # resample configuration
        self.interpolator_resample = configuration_resample.interpolator
        self.default_pixel_value = configuration_resample.default_pixel_value

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.result_transformation_path = Path()
        self.working_dir_path = Path()

        # results
        self.result_transformed_image = sitk.Image()
        self.result_transformation = sitk.Transform()

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):
        """
        Creates an affine transformation model to register the moving image to the fixed image.

        First the the similarity metric and optimizer are set.
        Then, the initial transform is set to an affine transform - this enforces ans affine transformation.
        Then the interpolator is set.

        The registration is executed and the result is stored (both the transformation and the transformed image).

        """

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path

        fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

        # store the folder where the images are stored
        self.working_dir_path = self.fixed_path.parent

        registration = sitk.ImageRegistrationMethod()

        self._set_similarity_metric(registration)

        self._set_optimizer(registration)

        # create and set affine initial transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration.SetInitialTransform(initial_transform)

        registration.SetInterpolator(self.interpolator_registration)

        if print_progress:
            registration.AddCommand(
                sitk.sitkIterationEvent, lambda: self._print_progress(registration))

        # register the images and store the result
        self.result_transformation = registration.Execute(
            fixed_image, moving_image)
        self.result_transformed_image = self._resample(
            fixed_image, moving_image)
        
        fixed_name = self.fixed_path.stem
        moving_name = self.moving_path.stem

        date_time = datetime.datetime.now()

        self.result_transformed_image_path = self.working_dir_path / \
            f"{moving_name}_warped_on_{fixed_name}_{self.method}_{date_time}.nii"
        self.result_transformation_path = self.working_dir_path / \
            f"{moving_name}_warped_on_{fixed_name}_{self.method}_{date_time}.tfm"
        
        sitk.WriteImage(self.result_transformed_image, self.result_transformed_image_path)
        sitk.WriteTransform(self.result_transformation, self.result_transformation_path)

        self._print_registratoin_result(registration)

    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation

        return self.result_transformation_path

    def _set_optimizer(self, registration):
        if self.optimizer == SITKOptimizer.REGULAR_STEP_GRADIENT_DESCENT:
            registration.SetOptimizerAsRegularStepGradientDescent(
                learningRate=self.learning_rate,
                minStep=self.min_step,
                numberOfIterations=self.number_of_iterations,
                gradientMagnitudeTolerance=self.gradient_magnitude_tolerance,
            )
            registration.SetOptimizerScalesFromIndexShift()
        else:
            raise ValueError("Invalid optimizer")

    def _set_similarity_metric(self, registration):
        if self.similarity_metric == SITKSimilarityMetric.NCC:
            registration.SetMetricAsCorrelation()
        elif self.similarity_metric == SITKSimilarityMetric.MATTES_MI:
            registration.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50)
        elif self.similarity_metric == SITKSimilarityMetric.MSE:
            registration.SetMetricAsMeanSquares()
        else:
            raise ValueError("Invalid similarity metric")

    def _print_registratoin_result(self, registration):
        print("-------")
        print(self.result_transformation)
        print(
            f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {registration.GetOptimizerIteration()}")
        print(f" Metric value: {registration.GetMetricValue()}")

    def _resample(self, fixed_image: sitk.Image, moving_image: sitk.Image):
        """
        Resample the moving image using the transformation.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(self.interpolator_resample)
        resampler.SetDefaultPixelValue(self.default_pixel_value)
        resampler.SetTransform(self.result_transformation)

        return resampler.Execute(moving_image)

    @ staticmethod
    def _print_progress(method):
        if method.GetOptimizerIteration() == 0:
            print("Estimated Scales: ", method.GetOptimizerScales())
        print(
            f"{method.GetOptimizerIteration():3} "
            + f"= {method.GetMetricValue():7.5f} "
            # + f": {method.GetOptimizerPosition()}"
        )
