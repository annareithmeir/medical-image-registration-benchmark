from enum import Enum


class TransformationType(Enum):
    """
    Enum for the transformation type.
    """
    NONE = 0
    RIGID = 1
    AFFINE = 2
    B_SPLINE = 3


class SITKSimilarityMetric(Enum):
    """
    Enum for the SITK similarity metric.
    """
    NCC = 1
    MATTES_MI = 2
    MSE = 3


class SITKOptimizer(Enum):
    """
    Enum for the SITK optimizer.
    """
    REGULAR_STEP_GRADIENT_DESCENT = 1
