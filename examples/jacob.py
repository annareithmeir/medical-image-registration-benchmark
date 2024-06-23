from time import time
from tqdm import tqdm
import matplotlib
from pathlib import Path
import sys
import os
import torch
import time
import SimpleITK as sitk

# THIS HAS TO BE BEFORE THE VOXELMORPH IMPORTS BECAUSE IN THE INITS MAGIC HAPPENS
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent / "latent_space_registration"))  # nopep8

from registrationbaselines.core import utils, utils_metrics, metrics  # nopep8
from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.evaluation.evaluation import Evaluation  # nopep8
from registrationbaselines.training.train_voxelmorph_feature import VoxelmorphFeatureTraining  # nopep8
from registrationbaselines.registration.voxelmorph import VoxelmorphReg  # nopep8
from registrationbaselines.registration.bspline_feature import BSplineFeature  # nopep8
from registrationbaselines.registration.bspline_niftyreg import BSplineNiftyReg  # nopep8

path = "/u/home/koeglf/Documents/code/registrationbaselines/tmp/results_paramsearch/BSplines_no_encoder_NCC_lr0.0005_reg[100000000]_it[1000]_sigma[[12, 12]]/deformations/patient102_frame01_deformation_to_patient102_frame13.pt"


a = metrics.displacement_field_metrics(torch.load(path).detach().cpu().numpy())

x = 0
