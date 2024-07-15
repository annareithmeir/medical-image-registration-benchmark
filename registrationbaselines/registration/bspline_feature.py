from pathlib import Path
import sys
from typing import Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image
import SimpleITK as sitk
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent))  # nopep8
sys.path.append(str(Path(__file__).parent.absolute().parent.parent.parent / "latent_space_registration"))  # nopep8

from registrationbaselines.registration._interface_registration import RegistrationInterface
from registrationbaselines.evaluation.evaluation import Evaluation
from registrationbaselines.data_loading import data_loaders

import latent_space_registration.airlab as al
import latent_space_registration.airlab.transformation as al_transformation
import latent_space_registration.airlab.loss as al_loss
import latent_space_registration.airlab.regulariser as al_regulariser


class BSplineFeature(RegistrationInterface):
    """
    Affine registration using NiftyReg.
    No default initialisation, as the choice of registration should be concious.
    """

    def __init__(self, configuration: Dict[str, Any]) -> None:

        self.configuration_all_params = configuration

        self.encoders = self.configuration_all_params["encoders"]
        self.metrics = self.configuration_all_params["metrics"]
        self.lrs = self.configuration_all_params["lrs"]
        self.regularisation_weights = self.configuration_all_params["regularisation_weights"]
        self.iterationss = self.configuration_all_params["iterationss"]
        self.sigmas = self.configuration_all_params["sigmas"]

        self.configuration = {}

        # paths
        self.fixed_path: Path
        self.moving_path: Path
        self.result_transformed_image_path: Path
        self.result_transformation_path: Path

    def register_all_parametr_sets(self, dataloader: Dataset):
        """
        Register all parameter sets.
        """

        for encoder in self.encoders:
            for metric in self.metrics:
                for lr in self.lrs:
                    for regularisation_weight in self.regularisation_weights:
                        for iterations in self.iterationss:
                            for sigma in self.sigmas:
                                self.configuration["encoder"] = encoder
                                self.configuration["metric"] = metric
                                self.configuration["lr"] = lr
                                self.configuration["regularisation_weight"] = regularisation_weight
                                self.configuration["iterations"] = iterations
                                self.configuration["sigma"] = sigma

                                self.method = self.configuration_all_params["method_name"] + \
                                    f"_{encoder}_{metric}_lr{lr}_reg{regularisation_weight}_it{iterations}_sigma{sigma}"
                                self._create_result_directories()

                                print("\nregister...")
                                for i in tqdm(range(len(dataloader))):
                                    item = dataloader[i]
                                    self.register(
                                        item["img_x"], item["img_y"])

                                # evaluate
                                loader_transformations = data_loaders.BaselineTransformations(
                                    Path(self.configuration_all_params["result_path"]) / self.method)

                                print("\nevaluate...")
                                evaluation = Evaluation(Path(self.configuration_all_params["result_path"]),
                                                        self.method)
                                evaluation.evaluate(
                                    loader_transformations, dataloader)
                                print("\nplot...")
                                evaluation.visualize(
                                    loader_transformations, dataloader)
                                print("\ndone")

    def register(self,
                 fixed_image: Tuple[Path, np.ndarray],
                 moving_image: Tuple[Path, np.ndarray]) -> None:
        """
        """

        use_rgb = False

        # check that both images exist
        assert fixed_image[0].exists(
        ), f"File {fixed_image[0]} does not exist."
        assert moving_image[0].exists(
        ), f"File {moving_image[0]} does not exist."

        self.fixed_path = fixed_image[0]
        self.moving_path = moving_image[0]

        dtype = torch.float32
        device = torch.device("cuda:0")

        image_fixed = fixed_image[1].squeeze()
        image_moving = moving_image[1].squeeze()

        image_fixed = al.utils.image_from_numpy(
            image_fixed, [1, 1], [0, 0], dtype=dtype, device=device)
        image_moving = al.utils.image_from_numpy(
            image_moving, [1, 1], [0, 0], dtype=dtype, device=device)

        image_fixed_pyramid = al.create_image_pyramid(image_fixed,
                                                      [[4, 4], [2, 2]])
        image_moving_pyramid = al.create_image_pyramid(image_moving,
                                                       [[4, 4], [2, 2]])

        for level, (image_fixed, image_moving) in enumerate(zip(image_fixed_pyramid, image_moving_pyramid)):

            regularisation_weight = self.configuration["regularisation_weight"][level]
            number_of_iterations = self.configuration["iterations"][level]
            sigma = self.configuration["sigma"][level]

            registration = al.PairwiseRegistration(verbose=True)

            # define the transformation
            # transformation = al.transformation.pairwise.RigidTransformation(
            #     image_moving)
            transformation = al_transformation.pairwise.BsplineTransformation(image_moving.size,
                                                                              sigma=sigma,
                                                                              rgb=use_rgb,
                                                                              order=1,
                                                                              dtype=dtype,
                                                                              device=device,
                                                                              diffeomorphic=True)

            if level > 0:
                constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                              image_moving.size,
                                                                              interpolation="linear")
                transformation.set_constant_flow(constant_flow)

            registration.set_transformation(transformation)

            # this means we do it on the images
            if self.configuration["encoder"] == "no_encoder":
                if self.configuration["metric"] == "MSE":
                    image_loss = al_loss.pairwise.MSE(
                        image_fixed, image_moving, rgb=use_rgb)
                elif self.configuration["metric"] == "NCC":
                    image_loss = al_loss.pairwise.NCC(
                        image_fixed, image_moving, rgb=use_rgb)
                else:
                    raise ValueError(
                        f'Metric {self.configuration["metric"]} not implemented')
            else:
                image_loss = al_loss.pairwise.LatentSpaceFeatureLoss(image_fixed,
                                                                     image_moving,
                                                                     rgb=use_rgb,
                                                                     extractor=self.configuration["encoder"],
                                                                     loss_type=self.configuration["metric"])

            registration.set_image_loss([image_loss])
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

            constant_flow = transformation.get_flow()

        # create final result
        displacement = transformation.get_displacement()
        warped_image = al_transformation.utils.warp_image(
            image_moving, displacement)

        self._save_results(warped_image, displacement)

    def _save_results(self, deformed: al.Image, deformation: torch.Tensor):
        self.result_transformed_image_path, \
            self.result_transformation_path = self._create_result_paths(self.fixed_path.stem,
                                                                        self.moving_path.stem,
                                                                        ".nii.gz",
                                                                        ".nii.gz")

        # SAVE DEFORMED IMAGE
        image_deformed = deformed.image.detach().cpu().numpy().squeeze()
        image_deformed = sitk.GetImageFromArray(image_deformed)
        image_deformed.SetSpacing(spacing=deformed.spacing)
        image_deformed.SetOrigin(origin=deformed.origin)
        sitk.WriteImage(image_deformed, self.result_transformed_image_path)

        # SAVE DEFORMATION
        deformation = al_transformation.utils.unit_displacement_to_displacement(
            deformation)
        itk_displacement = sitk.GetImageFromArray(
            deformation.detach().cpu().numpy(), isVector=True)
        itk_displacement.SetSpacing(spacing=deformed.spacing)
        itk_displacement.SetOrigin(origin=deformed.origin)
        sitk.WriteImage(itk_displacement, self.result_transformation_path)

        sitk.ImageReaderBase_GetImageIOFromFileName
