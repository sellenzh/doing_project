import inspect
import torch
import importlib
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kwargs):
        super().__init__()
        self.save_hyper_parameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
