from typing import Tuple, Union

import numpy as np


floatArray2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]

floatArray3D = np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]

floatArray2Dor3D = np.ndarray[Union[Tuple[int, int], Tuple[int, int, int]],
                              np.dtype[np.float64]]

floatArray3Dor4D = np.ndarray[Union[Tuple[int, int, int], Tuple[int, int, int, int]],
                              np.dtype[np.float64]]

floatArray2Dor3Dor4D = np.ndarray[Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]],
                                  np.dtype[np.float64]]

intArray2Dor3Dor4D = np.ndarray[Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]],
                                np.dtype[np.int32]]
