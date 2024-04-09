from enum import Enum

import SimpleITK as sitk


class TransformationType(Enum):
    """
    Enum for the transformation type.
    """
    AFFINE = 1
    NON_RIGID = 2


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
