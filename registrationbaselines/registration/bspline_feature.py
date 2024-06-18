from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
import SimpleITK as sitk
from registrationbaselines.registration._interface_registration import RegistrationInterface

sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent / "latent_space_registration"))  # nopep8


import latent_space_registration.airlab as al
import latent_space_registration.airlab.transformation as al_transformation
import latent_space_registration.airlab.loss as al_loss
import latent_space_registration.airlab.regulariser as al_regulariser


class BSplineFeature(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration_path: Path) -> None:

        self.configuration = self.read_config(configuration_path)

        self.method = self.configuration["method_name"]

        self._create_result_directories()

        # paths
        self.fixed_path: Path
        self.moving_path: Path
        self.result_transformed_image_path: Path
        self.result_transformation_path: Path

    def register(self,
                 fixed_image_path: Path,
                 moving_image_path: Path) -> None:
        """
        """

        # check that both images exist
        assert fixed_image_path.exists(
        ), f"File {fixed_image_path} does not exist."
        assert moving_image_path.exists(
        ), f"File {moving_image_path} does not exist."

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path

        dtype = torch.float32
        device = torch.device("cuda:0")

        image_fixed = np.array(Image.open(self.fixed_path))
        image_moving = np.array(Image.open(self.moving_path))

        image_fixed = np.moveaxis(image_fixed, -1, 0)
        image_moving = np.moveaxis(image_moving, -1, 0)

        image_fixed = al.utils.image_from_numpy(
            image_fixed, [1, 1], [0, 0], dtype=dtype, device=device)
        image_moving = al.utils.image_from_numpy(
            image_moving, [1, 1], [0, 0], dtype=dtype, device=device)

        regularisation_weight = self.configuration["regularisation_weight"]
        number_of_iterations = self.configuration["iterations"]

        sigma = self.configuration["sigma"]

        registration = al.PairwiseRegistration(verbose=True)

        # define the transformation
        # transformation = al.transformation.pairwise.RigidTransformation(
        #     image_moving)
        transformation = al_transformation.pairwise.BsplineTransformation(image_moving.size,
                                                                          sigma=sigma,
                                                                          rgb=True,
                                                                          order=1,
                                                                          dtype=dtype,
                                                                          device=device,
                                                                          diffeomorphic=True)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        # image_loss_image = al_loss.pairwise.MSE(
        #     image_fixed, image_moving, rgb=True)
        image_loss_feature = al_loss.pairwise.LatentSpaceFeatureLoss(image_fixed,
                                                                     image_moving,
                                                                     rgb=True,
                                                                     extractor=self.configuration["encoder"],
                                                                     loss_type=self.configuration["featureMetric"])

        # registration.set_image_loss([image_loss_image])
        registration.set_image_loss([image_loss_feature])
        # registration.set_image_loss(
        #     [image_loss_image, image_loss_feature], [0.5, 2])

        # define the regulariser for the displacement
        regulariser = al_regulariser.displacement.DiffusionRegulariser(
            image_moving.spacing)
        regulariser.SetWeight(regularisation_weight)
        registration.set_regulariser_displacement([regulariser])

        # define the optimizer
        optimizer = torch.optim.Adam(
            transformation.parameters(), lr=self.configuration["lr"])

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations)

        registration.start()

        # create final result
        displacement = transformation.get_displacement()
        warped_image = al_transformation.utils.warp_rgb_image(
            image_moving, displacement)

        warped_image.image = warped_image.image.permute(0, 1, 3, 4, 2)

        self._save_results(warped_image, displacement)

    def _save_results(self, deformed: al.Image, deformation: torch.Tensor):
        self.result_transformed_image_path, \
            self.result_transformation_path = self._create_result_paths(self.fixed_path.stem,
                                                                        self.moving_path.stem,
                                                                        ".jpg",
                                                                        ".nii.gz")

        image_deformed = deformed.image.detach().cpu().numpy().squeeze()
        image_deformed = Image.fromarray(image_deformed.astype(np.uint8))
        image_deformed.save(self.result_transformed_image_path)

        itk_displacement = sitk.GetImageFromArray(
            deformation.detach().cpu().numpy(), isVector=True)

        itk_displacement.SetSpacing(spacing=deformed.spacing)
        itk_displacement.SetOrigin(origin=deformed.origin)

        sitk.WriteImage(itk_displacement, self.result_transformation_path)
