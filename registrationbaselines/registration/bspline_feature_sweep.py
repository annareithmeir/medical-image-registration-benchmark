from pathlib import Path
import sys
from typing import Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image
import SimpleITK as sitk
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

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

    def __init__(self, configuration: Dict[str, Any],
                 configuration_register: Dict[str, Any]) -> None:

        self.configuration_wandb: Dict[str, Any] = configuration

        # this is used if you just want to use the register() function
        self.configuration_register: Dict[str, Any] = configuration_register

        # paths
        self.fixed_path: Path
        self.moving_path: Path
        self.result_transformed_image_path: Path
        self.result_transformation_path: Path

    def _register_wandb_wrapper(self, method_name: str):
        """
        Register all parameter sets.
        """

        # Set wandb to offline mode
        # IMPORTANT: this has to be called after creating wandb.agent() for some reason!
        wandb.init()

        joined_metric = ''.join(
            [f'{value}_{key}_' for key, value in wandb.config.metric.items()])

        method = method_name + \
            f"_{wandb.config.encoder}_" \
            f"{joined_metric}_" \
            f"lr{wandb.config.lr}_" \
            f"reg{wandb.config.regularisation_weight}_" \
            f"it{wandb.config.iterations}_" \
            f"sigma{wandb.config.sigma}"

        self._create_result_directories()

        print(f"\nregister with parameters: \n\
            encoder: {wandb.config.encoder}\n\
            metric: {joined_metric}\n\
            lr: {wandb.config.lr}\n\
            regularisation_weight: {wandb.config.regularisation_weight}\n\
            iterations: {wandb.config.iterations}\n\
            sigma: {wandb.config.sigma}\n")

        wandb.config.update({"metric_val": joined_metric})

        # assign wandb.config for register()
        self.configuration_register = wandb.config

        assert len(self.dataloader) > 0, "Dataloader is empty."
        for item in tqdm(self.dataloader):
            self.register(item["fixed_image"], item["moving_image"])

        # evaluate
        loader_transformations = data_loaders.BaselineTransformations(
            Path(self.configuration_register["result_path"]) / method)

        print("\nevaluate...")
        evaluation = Evaluation(Path(self.configuration_register["result_path"]),
                                method)
        evaluation.evaluate(
            loader_transformations, self.dataloader)

        # print("\nplot...")
        # evaluation.visualize(
        #     loader_transformations, self.dataloader)

        print("\nlog to wandb...")
        evaluation.wandb_log()

        print("done\n\n")

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

        regularisation_weight = self.configuration_register["regularisation_weight"]
        number_of_iterations = self.configuration_register["iterations"]
        sigma = self.configuration_register["sigma"]

        registration = al.PairwiseRegistration(verbose=False)

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

        registration.set_transformation(transformation)

        image_loss = []
        image_loss_weights = []
        # image losses
        if "MSE" in self.configuration_register["metric"]:
            image_loss.append(al_loss.pairwise.MSE(
                image_fixed, image_moving, rgb=use_rgb))
            image_loss_weights.append(
                self.configuration_register["metric"]["MSE"])

        if "NCC" in self.configuration_register["metric"]:
            image_loss.append(al_loss.pairwise.NCC(
                image_fixed, image_moving, rgb=use_rgb))
            image_loss_weights.append(
                self.configuration_register["metric"]["NCC"])

        if "MI" in self.configuration_register["metric"]:
            image_loss.append(al_loss.pairwise.MI(
                image_fixed, image_moving, rgb=use_rgb))
            image_loss_weights.append(
                self.configuration_register["metric"]["MI"])

        # feature losses
        if "COSINE" in self.configuration_register["metric"]:

            image_loss.append(al_loss.pairwise.LatentSpaceFeatureLoss(image_fixed,
                                                                      image_moving,
                                                                      rgb=use_rgb,
                                                                      extractor=self.configuration_register[
                                                                          "encoder"],
                                                                      loss_type="COSINE",
                                                                      dino_upsample_factor=0))
            image_loss_weights.append(
                self.configuration_register["metric"]["COSINE"])
        elif "L1" in self.configuration_register["metric"]:

            image_loss.append(al_loss.pairwise.LatentSpaceFeatureLoss(image_fixed,
                                                                      image_moving,
                                                                      rgb=use_rgb,
                                                                      extractor=self.configuration_register[
                                                                          "encoder"],
                                                                      loss_type="L1",
                                                                      dino_upsample_factor=0))
            image_loss_weights.append(
                self.configuration_register["metric"]["L1"])

        registration.set_image_loss(image_loss, image_loss_weights)

        # define the regulariser for the displacement
        regulariser = al_regulariser.displacement.DiffusionRegulariser(
            image_moving.spacing)
        regulariser.set_weight(regularisation_weight)
        registration.set_regulariser_displacement([regulariser])

        # define the optimizer
        optimizer = torch.optim.Adam(
            transformation.parameters(), lr=self.configuration_register["lr"])

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations)

        registration.start()

        # create final result
        displacement = transformation.get_displacement()
        warped_image = al_transformation.utils.warp_image(
            image_moving, displacement)

        self._save_results(warped_image, displacement, image_loss)

    def _save_results(self, deformed: al.Image, deformation: torch.Tensor, loss_lists: list):
        self.result_transformed_image_path, \
            self.result_transformation_path = self._create_result_paths(self.fixed_path.stem,
                                                                        self.moving_path.stem,
                                                                        ".pt",
                                                                        ".pt")

        # SAVE DEFORMED IMAGE
        torch.save(deformed.image, self.result_transformed_image_path)

        # SAVE DEFORMATION
        torch.save(deformation, self.result_transformation_path)

        # SAVE LOSS LISTS
        path_losses = self.result_transformation_path.parent.parent / "losses"
        path_losses.mkdir(exist_ok=True)

        for loss in loss_lists:

            name = loss._name

            loss_list_path = path_losses / \
                (str(self.result_transformation_path.name).replace(
                    ".pt", "") + f"_{name}_loss.png")

            # plot the loss
            loss_list = np.array(loss.loss_list)

            plt.figure()
            plt.plot(loss_list)
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.title(name)
            plt.savefig(loss_list_path.with_suffix(".png"))
            plt.ylim(bottom=-1.0)
