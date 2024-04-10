from pathlib import Path
import subprocess
import os

from typing import List

from ..core.transformation_interface import TransformationInterface
from ..core.configurations import AffineNiftyRegConfiguration
from ..core.enums import TransformationType
from ..core.utilities import create_result_paths

REG_TRANSFORM_PATH = Path(os.path.expanduser('~/bin/reg_transform'))


class TransformAffineNiftyReg(TransformationInterface)