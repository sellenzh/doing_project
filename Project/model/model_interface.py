import inspect
import torch
import importlib
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from sklearn.metrics import balanced_accuracy_score
import numpy as np

class MInterface(pl.LightningModule):
    def __init__(self, model_name, len, args, **kwargs):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.total_steps = len * args.epochs
        self.model_name = args.model_name

        self.save_hyper_parameters()
        self.load_model()
        #self.configure_loss()
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

        y_onehot = torch.FloatTensor(y.shape[0], 3).to(y.device).zero_()
        y_onehot.scatter_(1, y.long(), 1)
        loss = F.binary_cross_entropy_with_logits(out, y_onehot, weight=w)

        pred = out.softmax(1).argmax(1)
        acc = balanced_accuracy_score(pred.view(-1).long().cpu(), y.view())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc * 100.0, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pose = batch[0]
        y = batch[1]
        frame = batch[2]
        vel = batch[3]

        out = self(pose, frame, vel)
        w = self.val_weight

        y_onehot = torch.FloatTensor(y.shape[0], 3).to(y.device).zero_()
        y_onehot.scatter_(1, y.long(), 1)
        loss = F.binary_cross_entropy_with_logits(out, y_onehot, weight=w)

        pred = out.softmax(1).argmax(1)
        acc = balanced_accuracy_score(pred.view(-1).long().cpu(), y.view())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc * 100.0, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pose = batch[0]
        y = batch[1]
        frame = batch[2]
        vel = batch[3]

        out = self(pose, frame, vel)
        w = self.te_weight

        y_onehot = torch.FloatTensor(y.shape[0], 3).to(y.device).zero_()
        y_onehot.scatter_(1, y.long(), 1)
        loss = F.binary_cross_entropy_with_logits(out, y_onehot, weight=w)

        pred = out.softmax(1).argmax(1)
        acc = balanced_accuracy_score(pred.view(-1).long().cpu(), y.view())
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc * 100.0, prog_bar=True)
        return loss

    #def on_validation_epoch_end(self):

    def configure_optimizer(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = {'name': 'OneCycleLR', 'scheduler':
                        torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.lr, div_factor=10.0, total_steps=self.total_steps,
                                                            verbose=False),
                        'interval': 'step', 'frequency': 1, }
        return [optim], [lr_scheduler]

    def load_model(self):
        name = self.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')

        self.model = self.instancialize(model)

    def instancialize(self, model, **other_args):

        class_args = inspect.getfullargspec(model.__init__).args[1:]
        in_keys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in in_keys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return model(**args1)

