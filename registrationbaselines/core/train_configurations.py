class VoxelmorphTrainConfiguration():
    """ Configuration for voxelmorph training procedure"""

    def __init__(self,
                 batch_size: int = 1,
                 epochs: int = 1500,
                 steps_per_epoch: int = 100,
                 load_model: str = None,
                 initial_epoch: int = None,
                 lr: float = 1e-4,
                 int_steps: int = 7,
                 int_downsize: int = 2,
                 enc: list[int] = [16, 32, 32, 32],
                 dec: list[int] = [32, 32, 32, 32, 32, 16, 16],
                 bidir: bool = False,
                 sim_loss: str = "mse",
                 reg_weight: float = 0.01,
                 result_model_path: str = None,
                 use_wandb: bool = False,
                 gpu: str = '0',
                 save_checkpoint : int =20):
        """
        # todo docstring
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.load_model = load_model
        self.initial_epoch = initial_epoch
        self.lr = lr
        self.int_steps = int_steps
        self.int_downsize = int_downsize
        self.enc = enc
        self.dec = dec
        self.bidir = bidir
        self.sim_loss = sim_loss # 'ncc' or 'mse'
        self.reg_weight = reg_weight

        self.result_model_path = result_model_path
        self.use_wandb = use_wandb

        self.multi_channel = False
        self.cudnn_nondet = False
        self.gpu = gpu

        self.save_chackpoint = save_checkpoint
