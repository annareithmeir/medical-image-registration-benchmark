from pathlib import Path

import SimpleITK as sitk

from registrationbaselines.interfaces._interface_registration import RegistrationInterface


class AffineSITK(RegistrationInterface):
    """
    Affine registration using SimpleITK.
    No default initialisation, as the choice of registration and resampling should be concious.
    """

    def __init__(self,
                 configuration_path_registration: Path,
                 configuration_path_resample: Path) -> None:

        self.method = "AffneSITK"

        # registration configuration
        self.config_reg = self.read_config(configuration_path_registration)

        # resample configuration
        self.config_resample = self.read_config(configuration_path_resample)

        self._create_result_directories()

        # paths
        self.fixed_path = Path()
        self.moving_path = Path()
        self.result_transformed_image_path = Path()
        self.result_transformation_path = Path()
        self.working_dir_path = Path()

    def register(self, fixed_image_path: Path, moving_image_path: Path, print_progress: bool = False):
        """
        Creates an affine transformation model to register the moving image to the fixed image.

        First the the similarity metric and optimizer are set.
        Then, the initial transform is set to an affine transform - this enforces ans affine transformation.
        Then the interpolator is set.

        The registration is executed and the result is stored (both the transformation and the transformed image).

        """

        self.fixed_path = fixed_image_path
        self.moving_path = moving_image_path
        self.working_dir_path = self.fixed_path.parent

        # check that both images exist
        assert self.fixed_path.exists(
        ), f"File {self.fixed_path} does not exist."
        assert self.moving_path.exists(
        ), f"File {self.moving_path} does not exist."

        fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

        registration = self.__create_registration(
            fixed_image, moving_image, print_progress)

        # register the images
        result_transformation = registration.Execute(
            fixed_image, moving_image)
        result_transformed_image = self.__resample(
            result_transformation, fixed_image, moving_image)

        # save the results
        self._save_results(result_transformed_image, result_transformation)

        self.__print_registratoin_result(registration, result_transformation)

    def get_transformed_image_path(self):
        # Return transformed image
        return self.result_transformed_image_path

    def get_transformation_path(self):
        # Return transformation

        return self.result_transformation_path

    def _save_results(self, deformed, deformation):
        self.result_transformed_image_path, self.result_transformation_path = self._create_result_paths(self.fixed_path.stem,
                                                                                                        self.moving_path.stem,
                                                                                                        ".nii",
                                                                                                        ".tfm")

        sitk.WriteImage(deformed, self.result_transformed_image_path)
        sitk.WriteTransform(deformation, self.result_transformation_path)

    def __create_registration(self, fixed_image, moving_image, print_progress):
        registration = sitk.ImageRegistrationMethod()

        self.__set_similarity_metric(registration)
        self.__set_optimizer(registration)

        # create and set affine initial transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration.SetInitialTransform(initial_transform)

        if self.config_reg['interpolator'] == "sitkLinear":
            registration.SetInterpolator(sitk.sitkLinear)
        else:
            raise ValueError("Invalid interpolator")

        if print_progress:
            registration.AddCommand(
                sitk.sitkIterationEvent, lambda: self.__print_progress(registration))

        return registration

    def __set_optimizer(self, registration):
        if self.config_reg['optimiser'] == "regular_step_gradient_descent":
            registration.SetOptimizerAsRegularStepGradientDescent(
                learningRate=self.config_reg['learning_rate'],
                minStep=self.config_reg['min_step'],
                numberOfIterations=self.config_reg['number_of_iterations'],
                gradientMagnitudeTolerance=self.config_reg['gradient_magnitude_tolerance'],
            )
            registration.SetOptimizerScalesFromIndexShift()
        else:
            raise ValueError("Invalid optimizer")

    def __set_similarity_metric(self, registration):
        if self.config_reg['similarity_metric'] == "NCC":
            registration.SetMetricAsCorrelation()
        elif self.config_reg['similarity_metric'] == "MATTES_MI":
            registration.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50)
        elif self.config_reg['similarity_metric'] == "MSE":
            registration.SetMetricAsMeanSquares()
        else:
            raise ValueError("Invalid similarity metric")

    def __set_interpolator(self, sitk_object):
        if self.config_resample['interpolator'] == "sitkLinear":
            sitk_object.SetInterpolator(sitk.sitkLinear)
        elif self.config_resample['interpolator'] == "sitkHammingWindowedSinc":
            sitk_object.SetInterpolator(sitk.sitkHammingWindowedSinc)
        else:
            raise ValueError("Invalid interpolator")

    def __print_registratoin_result(self, registration, transformation):
        print("-------")
        print(transformation)
        print(
            f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {registration.GetOptimizerIteration()}")
        print(f" Metric value: {registration.GetMetricValue()}")

    def __resample(self, transformation, fixed_image: sitk.Image, moving_image: sitk.Image):
        """
        Resample the moving image using the transformation.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        self.__set_interpolator(resampler)
        resampler.SetDefaultPixelValue(
            self.config_resample['default_pixel_value'])
        resampler.SetTransform(transformation)

        return resampler.Execute(moving_image)

    @staticmethod
    def __print_progress(method):
        if method.GetOptimizerIteration() == 0:
            print("Estimated Scales: ", method.GetOptimizerScales())
        print(
            f"{method.GetOptimizerIteration():3} "
            + f"= {method.GetMetricValue():7.5f} "
            # + f": {method.GetOptimizerPosition()}"
        )
