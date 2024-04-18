This software provides the source-code for the algorithms described in

"Estimating Large Lung Motion in COPD Patients by Symmetric Regularised Correspondence Fields"
 by Mattias P. Heinrich, Heinz Handels and Ivor J.A. Simpson
 Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015, LNCS, Springer 

for estimating correspondence fields. For compilation it requires an installed zlib library (see http://zlib.net or use: "sudo apt-get install zlib1g-dev") and the eigenlibrary (see http://eigen.tuxfamily.org). The following line will build the code (with the I parameter you have to specigy where to look for the eigen library):
g++ -I /usr/include/eigen3 corrField.cpp -O3 -lpthread -std=c++11 -lz -msse4.2 -o corrField

The program only works with 3D nifti images (output will be nii.gz) and generates a 6 x num_keypoints array of correspondences (stored in float). applyCorrField will use a correspondences field to warp the moving image.

The scriptCOPD should replicate the experiments in the paper. You need to obtain the data from DIR-lab.com (free registration required) and download the employed segmentation masks from http://mpheinrich.de/copd_resampled_masks.zip . The exact case-by-case results with default settings are: 1.068, 1.570, 1.027, 1.035, 0.985, 1.063, 1.030, 1.085, 0.797, 1.203, avg.: 1.0863 mm. There are tiny differences compared to the paper due to some implementation choices, e.g. using Eigen3 for matrix inversions.

Let me know if you have any questions or find bugs. heinrich(at)uni-luebeck.de

Mattias
