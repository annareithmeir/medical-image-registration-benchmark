import icon_registration.itk_wrapper as itk_wrapper
import icon_registration.pretrained_models as pretrained_models
import itk


model = pretrained_models.OAI_knees_gradICON_model()  # GradICON
# model = pretrained_models.OAI_knees_registration_model() #ICON

# Feel free to experiment with different images here
image_A = itk.imread(
    "9487462_20081003_SAG_3D_DESS_RIGHT_11495603_image.nii.gz")
image_B = itk.imread(
    "9225063_20090413_SAG_3D_DESS_RIGHT_12784112_image.nii.gz")

phi_AB, phi_BA = itk_wrapper.register_pair(model, image_A, image_B)

model = pretrained_models.OAI_knees_gradICON_model()  # GradICON
# model = pretrained_models.OAI_knees_registration_model() #ICON

# Feel free to experiment with different images here
image_A = itk.imread(
    "9487462_20081003_SAG_3D_DESS_RIGHT_11495603_image.nii.gz")
image_B = itk.imread(
    "9225063_20090413_SAG_3D_DESS_RIGHT_12784112_image.nii.gz")

phi_AB, phi_BA = itk_wrapper.register_pair(model, image_A, image_B)
