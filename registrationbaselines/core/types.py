from pathlib import Path

from typing import Tuple, Dict, Union

import numpy as np
import torch


floatArray2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]

floatArray3D = np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]

floatArray2Dor3D = np.ndarray[Union[Tuple[int, int], Tuple[int, int, int]],
                              np.dtype[np.float64]]

floatArray3Dor4D = np.ndarray[Union[Tuple[int, int, int], Tuple[int, int, int, int]],
                              np.dtype[np.float64]]

floatArray2Dor3Dor4D = np.ndarray[Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]],
                                  np.dtype[np.float64]]

intArray2Dor3Dor4D = np.ndarray[Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]],
                                Union[np.dtype[np.uint8], np.dtype[np.uint16], np.dtype[np.uint32], np.dtype[np.uint64]]]

datasetReturnType = Dict[str, Union[Path,
                                    Union[floatArray2Dor3Dor4D,
                                          intArray2Dor3Dor4D],
                                    torch.Tensor]]
