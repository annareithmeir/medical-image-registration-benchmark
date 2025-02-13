from pathlib import Path
import sys
import time

from typing import Dict, Type, Union

import numpy as np
import os
# os.environ["WANDB_MODE"] = "disabled"

sys.path.append(str(Path(__file__).parent.absolute().parent))  # nopep8

from registrationbaselines.core import utils  # nopep8
from registrationbaselines.core.singleton_logger import SingletonLogger  # nopep8
from registrationbaselines.data_loading import data_loaders  # nopep8
from registrationbaselines.training.train_lapirn import LapIRN  # nopep8
from registrationbaselines.training.train_voxelmorph import VoxelMorph  # nopep8
from registrationbaselines.training.train_gradicon import GradICON  # nopep8

# os.environ["WANDB_MODE"] = "disabled"


def main() -> None:
    logger_instance = SingletonLogger(Path(__file__.replace(".py", ".log")),
                                      "RegBaselines")
    logger = logger_instance.get_logger()

    base_dir = Path(__file__).parent.parent.absolute()

    methods: Dict[Type[Union[VoxelMorph, LapIRN]], Path] = {
        # LapIRN: base_dir / "registrationbaselines/configs/LapIRN.yaml",
        # VoxelMorph: base_dir / "registrationbaselines/configs/VoxelMorph.yaml",
        GradICON: base_dir / "registrationbaselines/configs/GradICON.yaml",
    }

    idxs = np.arange(5)
    # np.random.shuffle(idxs)
    train_idx, val_idx = [0, 0, 0], [0, 0, 0]

    path_neckCT = Path(
        "/home/koeglf/data/preprocess_again/SerielleCTs_nii_forHumans/")
    path_lungCT = Path("/home/koeglf/data/LungCT/LungCT_preprcoessed")
    path_abdomenMRCT = Path("/data/AbdomenMRCT_preprocessed")

    datasets = [


        # (data_loaders.NeckCTDataset(dataset_path=path_neckCT,
        #                             indices=train_idx,
        #                             return_type="torch_tensor_dict"),
        #  data_loaders.NeckCTDataset(dataset_path=path_neckCT,
        #                             indices=val_idx,
        #                             return_type="torch_tensor_dict"))

        (data_loaders.L2RLungCTDataset(dataset_path=path_lungCT,
                                       indices=train_idx,
                                       return_type="torch_tensor_dict"),
         data_loaders.L2RLungCTDataset(dataset_path=path_lungCT,
                                       indices=val_idx,
                                       return_type="torch_tensor_dict"))
    ]

    # (data_loaders.L2RAbdominalMRCTDataset(dataset_path=path_abdomenMRCT,
    #                                       indices=train_idx,
    #                                       return_type="torch_tensor_dict"),
    #  data_loaders.L2RAbdominalMRCTDataset(dataset_path=path_abdomenMRCT,
    #                                       indices=val_idx,
    #                                       return_type="torch_tensor_dict")),

    # (data_loaders.ImagePairDataset(image_pairs=[
    #     [Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{train_idx[0]+1:02}_0000.nii.gz"),
    #      Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{train_idx[0]+1:02}_0001.nii.gz")],
    #     [Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{train_idx[1]+1:02}_0000.nii.gz"),
    #      Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{train_idx[1]+1:02}_0001.nii.gz")]
    # ],
    #     return_type="torch_tensor_dict"),
    #     data_loaders.ImagePairDataset(image_pairs=[
    #         [Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{val_idx[0]+1:02}_0000.nii.gz"),
    #          Path(f"/data/AbdomenMRCT_preprocessed/imagesTr/AbdomenMRCT_00{val_idx[0]+1:02}_0001.nii.gz")]
    #     ],
    #     return_type="torch_tensor_dict"))

    for train_dataset, val_dataset in datasets:

        for method, config_path in methods.items():

            training = method(train_dataset,
                              config_path,
                              val_dataset)

            # training.execute_with_one_parameter_set()
            training.perform_wandb_sweep(project_name="NeckCT_overfit")


if __name__ == "__main__":

    utils.turn_off_warnings()

    start_time = time.time()
    main()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


"""

            scaler = torch.amp.GradScaler()  # 🚀 Initialize AMP

            for step in range(self.run_configuration['steps_per_epoch']):
                torch.cuda.empty_cache()

                step_start_time = time.time()

                # generate inputs (and true outputs) and convert them to tensors
                with torch.amp.autocast():
                    inputs, y_true = next(generator)
                    inputs = [d.to(device).float() for d in inputs]
                    y_true = [d.to(device).float() for d in y_true]

                    # Should show multiple GPUs
                    print(f"Model is on: {next(model.parameters()).device}")

                    # run inputs through the model to produce a warped image and flow field
                    y_pred = model(*inputs)

                    # calculate total loss
                    loss = 0
                    loss_list = []
                    for n, loss_function in enumerate(losses):
                        curr_loss = loss_function(
                            y_true[n], y_pred[n]) * weights[n]
                        loss_list.append(curr_loss.item())
                        loss += curr_loss

                    epoch_loss.append(loss_list)
                    epoch_total_loss.append(loss.item())

                scaler.scale(loss).backward()  # 🚀 Scale gradients
                scaler.step(optimizer)
                scaler.update()
                # backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


"""


"""
8 workers - 8.46 GiB free
4 workers - 8.42 GiB free
2 workers - 530  MiB free
1 workers - 530  MiB free

"""
