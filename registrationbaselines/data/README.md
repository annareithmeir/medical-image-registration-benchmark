### Registation data for testing
tumor1.nii and tumor2.nii are taken from 3D Slicer's Sample Data module, they were originally named 'MRBrainTumor1' and 'MRBrainTumor2'
- they are not registered at all

then in affinely_registered_NiftyReg we have tumor2 affinely registered to tumor1
    and inside affinely_registered_NiftyReg in resampled_111 we have both affinely registered images resampled to isotropic 1x1x1 image space

### Mini-Training and Test Datasets
A mini training dataset of 3 image pairs and test datset of 1 image, landmarks and segmentations pair is provided. They are taken from the Learn2Reg LungCT dataset and are already preprocessed.