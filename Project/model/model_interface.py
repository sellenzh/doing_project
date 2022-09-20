import inspect
import torch
import importlib
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import numpy as np

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kwargs):
        super().__init__()
        self.save_hyper_parameters()
        self.load_model()
        self.configure_loss()
        np.random.seed(42)

        tr_num_samples = [9974, 5956, 7867]
        te_num_samples = [9921, 5346, 3700]
        val_num_samples = [3404, 1369, 1813]
        self.tr_weight = torch.from_numpy(np.min(tr_num_samples) / tr_num_samples).float().cuda()
        self.te_weight = torch.from_numpy(np.min(te_num_samples) / te_num_samples).float().cuda()
        self.val_weight = torch.from_numpy(np.min(val_num_samples) / val_num_samples).float().cuda()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        pose = batch[0]
        y = batch[1]
        frame = batch[2]
        vel = batch[3]

        if np.random.randint(10) >= 5:
            crop_size = np.random.randint(2, 21)
            pose = pose[:, :, -crop_size:]

        out = self(pose, frame, vel)
        w = self.tr_weight
        
        y_onehot = 
        loss = F.binary_cross_entropy_with_logits(out, y_onehot, weight=w)

