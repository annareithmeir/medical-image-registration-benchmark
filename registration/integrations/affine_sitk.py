import SimpleITK as sitk

from ..core.enums import SITKSimilarityMetric, SITKOptimizer, SITKInterpolator
from ..core.registration_interface import RegistrationInterface
from ..core.configurations import AffineSITKConfiguration


class AffineSITK(RegistrationInterface):
    """
    Affine registration using SimpleITK.
    """

    def __init__(self, configuration: AffineSITKConfiguration):

        self.similarity_metric = configuration.similarity_metric

        self.optimizer = configuration.optimizer
        self.learning_rate = configuration.learning_rate
        self.min_step = configuration.min_step
        self.number_of_iterations = configuration.number_of_iterations
        self.gradient_magnitude_tolerance = configuration.gradient_magnitude_tolerance

        self.interpolator = configuration.interpolator

        self.transformation = None

    def register(self, fixed_image: sitk.Image, moving_image: sitk.Image, print_progress=False):
        registration = sitk.ImageRegistrationMethod()

        if self.similarity_metric == SITKSimilarityMetric.NCC:
            registration.SetMetricAsCorrelation()
        elif self.similarity_metric == SITKSimilarityMetric.MATTES_MI:
            registration.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50)
        elif self.similarity_metric == SITKSimilarityMetric.MSE:
            registration.SetMetricAsMeanSquares()
        else:
            raise ValueError("Invalid similarity metric")

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

        # affine initial transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)

        registration.SetInitialTransform(initial_transform)

        if self.interpolator == SITKInterpolator.LINEAR:
            registration.SetInterpolator(sitk.sitkLinear)
        elif self.interpolator == SITKInterpolator.BSPLINE:
            registration.SetInterpolator(sitk.sitkBSpline)
        else:
            raise ValueError("Invalid interpolator")

        if print_progress:
            registration.AddCommand(
                sitk.sitkIterationEvent, lambda: self._print_progress(registration))

        self.transformation = registration.Execute(fixed_image, moving_image)

        print("-------")
        print(self.transformation)
        print(
            f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {registration.GetOptimizerIteration()}")
        print(f" Metric value: {registration.GetMetricValue()}")

    def get_transformation(self):
        # Return transformation model

        return self.transformation

    @staticmethod
    def _print_progress(method):
        if method.GetOptimizerIteration() == 0:
            print("Estimated Scales: ", method.GetOptimizerScales())
        print(
            f"{method.GetOptimizerIteration():3} "
            + f"= {method.GetMetricValue():7.5f} "
            + f": {method.GetOptimizerPosition()}"
        )
