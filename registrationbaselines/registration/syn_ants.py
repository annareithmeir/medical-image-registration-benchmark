import ants
from pathlib import Path

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.core import utils_nifti


class SyNANTs(RegistrationInterface):
    """
    SyN registration using ANTs.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration_path: Path) -> None:
        """
        Initialize the registration model.

        NOTE: the displacement field won't work in slicer correctly if the correct itent code is set - the original should be left.
        """

        self.method_name = "SyNANTs"

        self.base_dir = Path(__file__).parent.parent.absolute().parent

        self.configuration = self.read_config(self.base_dir / configuration_path)

        self._create_result_directories()

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.result_transformation_path = Path()

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):
        """
        Wrapper around ants to register.
        """

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path

        # check that both images exist
        assert self.fixed_path.exists(
        ), f"File {self.fixed_path} does not exist."
        assert self.moving_path.exists(
        ), f"File {self.moving_path} does not exist."

        # load boath images with ants
        fixed_image = ants.image_read(self.fixed_path.as_posix())
        moving_image = ants.image_read(self.moving_path.as_posix())

        # Perform registration
        registration = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='SyNOnly',
            write_composite_transform=True  # nopep8 this outputs one .h5 transform, otherwise we have a .nii.gz and .mat
        )

        self._create_result_directories()

        self._save_results(
            registration['warpedmovout'], registration['fwdtransforms'])

    def _register_wandb_wrapper(self) -> None:
        pass

    def get_transformation_path(self):
        return self.result_transformation_path

    def get_transformed_image_path(self):
        return self.result_transformed_image_path

    def _save_results(self, deformed, deformation):
        self.result_transformed_image_path, self.result_transformation_path = \
            self._create_result_paths(self.fixed_path.stem,
                                      self.moving_path.stem,
                                      ".nii.gz",
                                      ".nii.gz")

        # save transformation (by converting to .nii.gz)
        utils_nifti.convert_h5_to_nii(self.fixed_path,
                                      Path(deformation),
                                      self.result_transformation_path)
        utils_nifti.set_intent_code(self.result_transformation_path, "NIFTI_INTENT_DISPVECT")

        # save transformed image
        deformed.to_filename(self.result_transformed_image_path)
