from abc import ABC, abstractmethod

import SimpleITK as sitk

from .enums import SITKSimilarityMetric, SITKOptimizer, SITKInterpolator


class RegistrationConfiguration(ABC):
    """
    Abstract base class for registration configurations.
    """
    @abstractmethod
    def __init__(self):
        """
        Initialize the registration model.
        """


class AffineSITKConfiguration(RegistrationConfiguration):
    """Configuration for the affine registration using SimpleITK"""

    def __init__(self, similarity_metric: SITKSimilarityMetric = SITKSimilarityMetric.NCC,
                 optimizer: SITKOptimizer = SITKOptimizer.REGULAR_STEP_GRADIENT_DESCENT,
                 learning_rate: float = 2.0,
                 min_step: float = 1e-4,
                 number_of_iterations: int = 500,
                 gradient_magnitude_tolerance: float = 1e-8,
                 interpolator: SITKInterpolator = sitk.sitkLinear):

        self.similarity_metric = similarity_metric

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_step = min_step
        self.number_of_iterations = number_of_iterations
        self.gradient_magnitude_tolerance = gradient_magnitude_tolerance

        self.interpolator = interpolator
