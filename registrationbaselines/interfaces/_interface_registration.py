from pathlib import Path
import shutil
import traceback

from abc import abstractmethod
from typing import Optional, Tuple

import torch
import wandb
from tqdm import tqdm

from registrationbaselines.core import utils_wandb, utils
from registrationbaselines.data_loading import data_loaders
from registrationbaselines.displacement import deform_objects
from registrationbaselines.evaluation.evaluation import RegistrationEvaluator
from registrationbaselines.interfaces import _interface_core
from registrationbaselines.io import save, load


class RegistrationInterface(_interface_core.InterfaceCore):
    """
    Abstract base class for registration models.
    """

    dataloader: data_loaders.GenericDataset

    path_dir_deformed: Path
    path_dir_deformations: Path

    path_result_deformation: Path
    path_result_deformed: Path

    evaluator: RegistrationEvaluator
    use_masked_evaluation: bool

    def __init__(self,
                 method_name: str,
                 configuration_path: Path,
                 path_results: Path,
                 dataloader: data_loaders.GenericDataset,
                 use_masked_evaluation: bool = True,
                 model_path: Optional[Path] = None,
                 use_logger: bool = False) -> None:
        """
        Initialize the registration model.
        """

        super().__init__(method_name,
                         configuration_path,
                         path_results,
                         dataloader.name,
                         use_logger)

        self.dataloader = dataloader
        self.use_masked_evaluation = use_masked_evaluation

        self.model_path = model_path

    @abstractmethod
    def _register(self,
                  fixed_image: torch.Tensor,
                  moving_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Register moving_image to fixed_image.

        This function has to call _save_results() at the end.

        @param fixed_image: fixed image
        @param moving_image: moving image

        @return: warped image, displacement field
        """

    def reevaluate_one_run(self, path_dir_run: Path = Path("/home/koeglf/data/registrationStudy/SerielleCTs_nii_forHumans_registrations/BSplineNiftyReg/BSplineNiftyReg_662d4caf-b56e-48a9-8803-4e8912161d8c")) -> None:
        self.evaluator = RegistrationEvaluator(path_dir_run,
                                               len(self.dataloader))
        self.log('reevaluate')
        for item in tqdm(self.dataloader):

            fixed_image = load.load_image(item["fixed_image"]).to(self.device)
            moving_image = load.load_image(
                item["moving_image"]).to(self.device)

            if self.dataloader.has_segmentations:
                fixed_segmentations = load.load_segmentation(
                    item["fixed_segmentations"]).to(self.device)
                moving_segmentations = load.load_segmentation(
                    item["moving_segmentations"]).to(self.device)

                fixed_segmentations_name = str(
                    item["fixed_segmentations"].stem).split('.')[0]
                moving_segmentations_name = str(
                    item["moving_segmentations"].stem).split('.')[0]
            else:
                fixed_segmentations, moving_segmentations = None, None
                fixed_segmentations_name, moving_segmentations_name = "", ""

            if self.use_masked_evaluation:
                fixed_evaluation_mask = utils.get_convex_hull_mask(fixed_image)
            else:
                fixed_evaluation_mask = None

            fixed_image_name = str(item["fixed_image"].stem).split('.')[0]
            moving_image_name = str(item["moving_image"].stem).split('.')[0]

            deformation_path = path_dir_run / "deformations" / \
                f"{moving_image_name}_deformation_to_{fixed_image_name}.nii.gz"

            displacement = load.load_displacement(
                deformation_path).to(self.device)

            own_warped = deform_objects.deform_image(
                moving_image, displacement)

            self.evaluator.evaluate(fixed_image_name,
                                    displacement.detach().clone(),
                                    (fixed_segmentations, fixed_segmentations_name),
                                    (moving_segmentations,
                                     moving_segmentations_name),
                                    fixed_evaluation_mask)

            self.evaluator.visualize(fixed_image,
                                     fixed_image_name,
                                     moving_image,
                                     moving_image_name,
                                     own_warped,
                                     displacement.detach().clone(),
                                     fixed_segmentations,
                                     moving_segmentations,
                                     fixed_evaluation_mask)

        self.evaluator.results.calculate_all_statistics()

    def execute_with_one_parameter_set(self) -> None:
        """
        Register and evaluate all files and log to wand.

        @return: None
        """

        if self.model_path and self.use_wandb:
            wandb.finish()
            raise ValueError(
                "DL mode can't be used with wandb sweeps for registration, because the sweep was done in training.")

        self._create_run_parameters()

        self._create_run_directory()

        if not self.model_path:
            self._save_run_configuration()

        self.evaluator = RegistrationEvaluator(self.path_dir_run,
                                               len(self.dataloader))
        for item in tqdm(self.dataloader):
            try:
                # if '0e5fp8GltvE' not in str(item["fixed_image"]):
                #     continue

                self.log(f"Registering {str(item['fixed_image']).split('/')[-1]} to {str(item['moving_image']).split('/')[-1]}")  # nopep8

                self.log(f"\tLoading images")
                fixed_image = load.load_image(
                    item["fixed_image"]).to(self.device)
                moving_image = load.load_image(
                    item["moving_image"]).to(self.device)

                if self.dataloader.has_segmentations:
                    self.log(f"\tLoading segmentations")
                    fixed_segmentations = load.load_segmentation(
                        item["fixed_segmentations"]).to(self.device)
                    moving_segmentations = load.load_segmentation(
                        item["moving_segmentations"]).to(self.device)

                    fixed_segmentations_name = str(
                        item["fixed_segmentations"].stem).split('.')[0]
                    moving_segmentations_name = str(
                        item["moving_segmentations"].stem).split('.')[0]
                else:
                    fixed_segmentations, moving_segmentations = None, None
                    fixed_segmentations_name, moving_segmentations_name = "", ""

                if self.use_masked_evaluation:
                    self.log(f"\tCreating evaluation mask")
                    fixed_evaluation_mask = utils.get_convex_hull_mask(
                        fixed_image)
                else:
                    fixed_evaluation_mask = None

                fixed_image_name = str(item["fixed_image"].stem).split('.')[0]
                moving_image_name = str(
                    item["moving_image"].stem).split('.')[0]

                self.log(f"\tRegistering")
                deformed_image, displacement = self._register(fixed_image,
                                                              moving_image)

                deformed_image = deformed_image.to(self.device)
                displacement = displacement.to(self.device)

                self.log(f"\tSaving results")
                self._save_results(deformed_image,
                                   displacement.detach().clone(),
                                   fixed_image_name,
                                   moving_image_name)

                self.log(f"\tEvaluating results")
                own_warped = deform_objects.deform_image(
                    moving_image, displacement)

                self.evaluator.evaluate(fixed_image_name,
                                        displacement.detach().clone(),
                                        (fixed_segmentations,
                                         fixed_segmentations_name),
                                        (moving_segmentations,
                                         moving_segmentations_name),
                                        fixed_evaluation_mask)

                self.log(f"\tVisualising results")
                self.evaluator.visualize(fixed_image,
                                         fixed_image_name,
                                         moving_image,
                                         moving_image_name,
                                         own_warped,
                                         displacement.detach().clone(),
                                         fixed_segmentations,
                                         moving_segmentations,
                                         fixed_evaluation_mask)
            except Exception as e:
                error_details = traceback.format_exc()
                self.log(f"Error in {fixed_image_name} to {moving_image_name}: {error_details}")  # nopep8

        self.log(f"\tCalculating statistics")
        self.evaluator.results.calculate_all_statistics()

        if self.use_wandb:
            self.log(f"\tLogging to wandb")
            self.evaluator.wandb_log()

        self.log(f"\tFinished")

    def evaluate_with_zero_displacement(self) -> None:
        """
        Evaluate with zero displacement.

        @return: None
        """

        self.run_name = "_zeroDisplacement"

        if self.use_wandb:
            self.run_configuration = wandb.config
        else:
            self.run_configuration = utils_wandb.convert_to_non_wandb_config(
                self.general_configuration)

        self._create_run_directory()
        self.path_dir_deformations.rmdir()
        self.path_dir_deformed.rmdir()

        self.evaluator = RegistrationEvaluator(self.path_dir_run,
                                               len(self.dataloader))

        # if a zero displacement evaluation is already present, raise
        if (self.path_dir_dataset / self.path_dir_run.name).exists():
            raise FileExistsError(
                f"Directory {self.path_dir_run} already exists.")

        for item in tqdm(self.dataloader):

            fixed_image = load.load_image(item["fixed_image"]).to(self.device)
            moving_image = load.load_image(
                item["moving_image"]).to(self.device)

            if self.dataloader.has_segmentations:
                fixed_segmentations = load.load_segmentation(
                    item["fixed_segmentations"]).to(self.device)
                moving_segmentations = load.load_segmentation(
                    item["moving_segmentations"]).to(self.device)
            else:
                fixed_segmentations, moving_segmentations = None, None

            if self.use_masked_evaluation:
                fixed_evaluation_mask = utils.get_convex_hull_mask(fixed_image)
            else:
                fixed_evaluation_mask = None

            fixed_image_name = str(item["fixed_image"].stem).split('.')[0]
            moving_image_name = str(item["moving_image"].stem).split('.')[0]
            fixed_segmentations_name = str(
                item["fixed_segmentations"].stem).split('.')[0]
            moving_segmentations_name = str(
                item["moving_segmentations"].stem).split('.')[0]

            self.evaluator.evaluate(row_name=fixed_image_name,
                                    fixed_segmentations=(
                                        fixed_segmentations, fixed_segmentations_name),
                                    moving_segmentations=(moving_segmentations,
                                                          moving_segmentations_name),
                                    fixed_evaluation_mask=fixed_evaluation_mask)

            self.evaluator.visualize(fixed_image,
                                     fixed_image_name,
                                     moving_image,
                                     moving_image_name,
                                     fixed_segmentations=fixed_segmentations,
                                     moving_segmentations=moving_segmentations,
                                     fixed_evaluation_mask=fixed_evaluation_mask)

        self.evaluator.results.calculate_all_statistics()

        # move the results directory one level up
        shutil.move(self.path_dir_run, self.path_dir_dataset)

        # if the path_dir_method_is empty, delete it
        if not list(self.path_dir_method.iterdir()):
            self.path_dir_method.rmdir()

    def get_transformation_path(self):
        """
        Return the transformation model.
        """
        return self.path_result_deformation

    def get_transformed_image_path(self):
        """
        Return the transformed image.
        """
        return self.path_result_deformed

    def _save_results(self,
                      deformed: torch.Tensor,
                      deformation: torch.Tensor,
                      fixed_image_name: str,
                      moving_image_name: str) -> None:
        """
        Save the results of the registration.
        """

        self.path_result_deformed, \
            self.path_result_deformation = self._create_result_paths(fixed_image_name,
                                                                     moving_image_name,
                                                                     ".nii.gz",
                                                                     ".nii.gz")

        # SAVE DEFORMED IMAGE
        save.save_image(deformed,
                        self.path_result_deformed)

        # SAVE DEFORMATION
        save.save_displacement(deformation,
                               self.path_result_deformation)

        if not self.path_result_deformed.exists():
            raise FileNotFoundError(
                f"File {self.path_result_deformed} couldn't be saved.")
        if not self.path_result_deformation.exists():
            raise FileNotFoundError(
                f"File {self.path_result_deformation} couldn't be saved.")

    def _create_result_paths(self,
                             name_fixed: str,
                             name_moving: str,
                             extension_image: str,
                             extension_transformation: str):
        """
        Create the paths for the result files (warped image and transformation).
        """

        name_moving = name_moving.replace(".nii", "")
        name_fixed = name_fixed.replace(".nii", "")

        name_moving = name_moving.replace(".gz", "")
        name_fixed = name_fixed.replace(".gz", "")

        path_dir_deformed = self.path_dir_deformed / \
            f"{name_moving}_deformed_to_{name_fixed}"
        path_deformation = self.path_dir_deformations / \
            f"{name_moving}_deformation_to_{name_fixed}"

        path_dir_deformed = Path(
            path_dir_deformed.as_posix() + extension_image)
        path_deformation = Path(
            path_deformation.as_posix() + extension_transformation)

        return Path(path_dir_deformed), Path(path_deformation)

    def _create_run_directory(self) -> None:
        """
        Create the run directory in the method directory.
        """

        self.path_dir_run = self.path_dir_method / self.run_name
        self.path_dir_run.mkdir(parents=True, exist_ok=True)

        # create two subdirectories 'deformed' and 'deformations'
        self.path_dir_deformed = self.path_dir_run / 'deformed'
        self.path_dir_deformed.mkdir(parents=True, exist_ok=True)

        self.path_dir_deformations = self.path_dir_run / 'deformations'
        self.path_dir_deformations.mkdir(parents=True, exist_ok=True)

        if not self.path_dir_deformations.exists():
            raise FileNotFoundError(
                f"Directory {self.path_dir_deformations} couldn't be created.")
        if not self.path_dir_deformed.exists():
            raise FileNotFoundError(
                f"Directory {self.path_dir_deformed} couldn't be created.")
