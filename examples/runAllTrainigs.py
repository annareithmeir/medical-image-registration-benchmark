from pathlib import Path
import sys

from typing import Dict, Type, Union

import numpy as np

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.training.train_voxelmorph import VoxelMorph
from registrationbaselines.training.train_lapirn import LapIRN
from registrationbaselines.data_loading import data_loaders


def main() -> None:

    base_dir = Path(__file__).parent.parent.absolute()

    methods: Dict[Type[Union[VoxelMorph, LapIRN]], Path] = {
        VoxelMorph: base_dir / "registrationbaselines/configs/VoxelMorph.yaml",
        # LapIRN: base_dir / "registrationbaselines/configs/LapIRN.yaml",
    }

    idxs = np.arange(5)
    np.random.shuffle(idxs)
    train_idx, val_idx = list(idxs[:4]), list(idxs[4:])

    path_lungCT = Path("/data/LungCT_preprocessed")
    path_abdomenMRCT = Path("/data/AbdomenMRCT_preprocessed")

    datasets = [
        (data_loaders.L2RLungCTDataset(dataset_path=path_lungCT,
                                       indices=train_idx,
                                       return_type="torch_tensor_dict"),
         data_loaders.L2RLungCTDataset(dataset_path=path_lungCT,
                                       indices=val_idx,
                                       return_type="torch_tensor_dict")),


        (data_loaders.L2RAbdominalMRCTDataset(dataset_path=path_abdomenMRCT,
                                              indices=train_idx,
                                              return_type="torch_tensor_dict"),
         data_loaders.L2RAbdominalMRCTDataset(dataset_path=path_abdomenMRCT,
                                              indices=val_idx,
                                              return_type="torch_tensor_dict")),

        (data_loaders.ImagePairDataset(image_pairs=[
            [Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{train_idx[0]:02}_0000.nii.gz"),
             Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{train_idx[0]:02}_0001.nii.gz")],
            [Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{train_idx[1]:02}_0000.nii.gz"),
             Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{train_idx[1]:02}_0001.nii.gz")]
        ],
            return_type="torch_tensor_dict"),
            data_loaders.ImagePairDataset(image_pairs=[
                [Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{val_idx[0]:02}_0000.nii.gz"),
                 Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{val_idx[0]:02}_0001.nii.gz")]
            ],
            return_type="torch_tensor_dict"))
    ]

    for dataset in datasets:
        train_dataset, val_dataset = dataset

        for method, config_path in methods.items():

            training = method(train_dataset,
                              config_path,
                              val_dataset)

            training.execute_with_one_parameter_set()

            training.perform_wandb_sweep()


if __name__ == "__main__":
    main()
