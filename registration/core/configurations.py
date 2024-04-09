import SimpleITK as sitk

from .enums import SITKSimilarityMetric, SITKOptimizer, TransformationType


class AffineSITKConfiguration():
    """Configuration for the affine registration using SimpleITK"""

    def __init__(self, similarity_metric: SITKSimilarityMetric = SITKSimilarityMetric.NCC,
                 optimizer: SITKOptimizer = SITKOptimizer.REGULAR_STEP_GRADIENT_DESCENT,
                 learning_rate: float = 2.0,
                 min_step: float = 1e-4,
                 number_of_iterations: int = 500,
                 gradient_magnitude_tolerance: float = 1e-8,
                 interpolator=sitk.sitkLinear):

        self.similarity_metric = similarity_metric

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_step = min_step
        self.number_of_iterations = number_of_iterations
        self.gradient_magnitude_tolerance = gradient_magnitude_tolerance

        self.interpolator = interpolator


class ResampleSITKConfiguration():
    """Configuration for resampling using SimpleITK"""

    def __init__(self, interpolator=sitk.sitkHammingWindowedSinc,
                 default_pixel_value: float = 100):

        self.interpolator = interpolator
        self.default_pixel_value = default_pixel_value


class AffineNiftyRegConfiguration():
    """Configuration for the affine registration using NiftyReg"""

    def __init__(self, transformation_type: TransformationType = TransformationType.AFFINE) -> None:
        self.transformation_type = transformation_type


class BSplineNiftyRegConfiguration():
    """Configuration for the affine registration using NiftyReg"""

    def __init__(self, transformation_type: TransformationType = TransformationType.B_SPLINE) -> None:
        self.transformation_type = transformation_type
