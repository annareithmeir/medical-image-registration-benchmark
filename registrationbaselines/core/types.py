from pathlib import Path

from typing import Tuple, Dict, Union

import numpy as np
import torch


allInts = Union[np.dtype[np.uint8], np.dtype[np.uint16],
                np.dtype[np.uint32], np.dtype[np.uint64],
                np.dtype[np.int8], np.dtype[np.int16],
                np.dtype[np.int32], np.dtype[np.int64]]

intArray1D = np.ndarray[Tuple[int],
                        allInts]

floatArray2D = np.ndarray[Tuple[int, int],
                          np.dtype[np.float64]]
intArray2D = np.ndarray[Tuple[int, int],
                        allInts]
array2D = Union[floatArray2D, intArray2D]

floatArray3D = np.ndarray[Tuple[int, int, int],
                          np.dtype[np.float64]]
intArray3D = np.ndarray[Tuple[int, int, int],
                        allInts]
array3D = Union[floatArray3D, intArray3D]

floatArray4D = np.ndarray[Tuple[int, int, int, int],
                          np.dtype[np.float64]]
intArray4D = np.ndarray[Tuple[int, int, int, int],
                        allInts]
array4D = Union[floatArray4D, intArray4D]

floatArray5D = np.ndarray[Tuple[int, int, int, int, int],
                          np.dtype[np.float64]]

floatArray2Dor3D = Union[floatArray2D, floatArray3D]
floatArray3Dor4D = Union[floatArray3D, floatArray4D]
floarArray4Dor5D = Union[floatArray4D, floatArray5D]
floatArray2Dor3Dor4D = Union[floatArray2Dor3D, floatArray3D, floatArray4D]


intArray2Dor3D = Union[intArray2D, intArray3D]
intArray3Dor4D = Union[intArray3D, intArray4D]
intArray2Dor3Dor4D = Union[intArray2D, intArray3D, intArray4D]

array2Dor3D = Union[floatArray2Dor3D, intArray2Dor3D]
array3Dor4D = Union[floatArray3Dor4D, intArray3Dor4D]

datasetReturnType = Dict[str, Union[Path,
                                    Union[floatArray2Dor3Dor4D,
                                          intArray2Dor3Dor4D],
                                    torch.Tensor]]
