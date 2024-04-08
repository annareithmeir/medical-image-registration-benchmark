from enum import Enum


class TransformationType(Enum):
    AFFINE = 1
    NON_RIGID = 2


class SITKSimilarityMetric(Enum):
    NCC = 1
    MATTES_MI = 2
    MSE = 3


class SITKOptimizer(Enum):
    REGULAR_STEP_GRADIENT_DESCENT = 1
