import torch
from torch.nn import functional as F
from torchvision import transforms as A
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import argparse
from pathlib import Path

from dataloader22 import DataSet
from models.model_jaad import Model

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.functional.classification.accuracy import accuracy

def seed_all(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class LitPedGraph(pl.LightningModule):
    def __init__(self, args, len_tr):
        super(LitPedGraph, self).__init__()
        self.balance = args.balance
        self.total_steps = len_tr * args.epochs
        self.lr = args.lr
        self.epochs = args.epochs
        self.model = Model(args)
        device = self.model.linear.weight.device
        tr_num_samples = [1025, 4778, 17582]
        self.tr_weight = torch.from_numpy(np.min(tr_num_samples) / tr_num_samples).float().to(device)
        te_num_samples = [1871, 3204, 13037]
        self.te_weight = torch.from_numpy(np.min(te_num_samples) / te_num_samples).float().to(device)
        val_num_samples = [176, 454, 2772]
        self.val_weight = torch.from_numpy(np.min(val_num_samples) / val_num_samples).float().to(device)

    def forward(self, kps, img, bbox, vel):
        return self.model(kps, img, bbox, vel)
    
    def training_step(self, batch, batch_nb):
        kps = batch[0]
        label = batch[1]
        img = batch[2]
        bbox = batch[3]
        vel = batch[4]
        logits = self(kps, img, bbox, vel)
        w = None if self.balance else self.tr_weight

        y_onehot = torch.FloatTensor(label.shape[0], 2).to(label.device).zero_()
        y_onehot.scatter_(1, label.long(), 1)
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot, weight=w)
        preds = logits.softmax(1).argmax(1)
        acc = accuracy(preds.view(-1).long(), label.view(-1).long())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc * 100.0, prog_bar=True)
        return loss

    def validation_steps(self, batch, batch_nb):
        kps = batch[0]
        label = batch[1]
        img = batch[2]
        bbox = batch[3]
        vel = batch[4]
        logits = self(kps, img, bbox, vel)
        w = None if self.balance else self.val_weight
        
        y_onehot = torch.FloatTensor(label.shape[0], 2).to(label.device).zero_()
        y_onehot.scatter_(1, label.long(), 1)
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot, weight=w)

        preds = logits.softmax(1).argmax(1) 
        acc = accuracy(preds.view(-1).long(), label.view(-1).long())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc * 100.0, prog_bar=True)
        return loss

    def test_steps(self, batch, batch_nb):
        kps = batch[0]
        label = batch[1]
        img = batch[2]
        bbox = batch[3]
        vel = batch[4]
        logits = self(kps, img, bbox, vel)
        w = None if self.balance else self.te_weight
        # loss = F.cross_entropy(logits, y.view(-1).long(), weight=w)
        
        y_onehot = torch.FloatTensor(label.shape[0], 2).to(label.device).zero_()
        y_onehot.scatter_(1, label.long(), 1)
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot, weight=w)

        preds = logits.softmax(1).argmax(1)
        acc = accuracy(preds.view(-1).long(), label.view(-1).long())
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc * 100.0, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        lr_scheduler = {
            'name': 'OneCycleLR',
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optim, max_lr=self.lr, div_factor=10.0, final_div_factor=1e4, total_steps=self.total_steps, verbose=False),
            'interval': 'step',
            'frequency': 1}
        return [optim], [lr_scheduler]

def data_loader(args):
    transform = A.Compose([A.ToPILImage(),
        A.RandomPosterize(bits=2),
        A.RandomInvert(p=0.2),
        A.RandomSolarize(threshold=50.0),
        A.RandomAdjustSharpness(sharpness_factor=2),
        A.RandomAutocontrast(p=0.2),
        A.RandomEqualize(p=0.2),
        A.ColorJitter(0.5, 0.3),
        A.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)), 
        A.ToTensor(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    tr_data = DataSet(path=args.data_path, jaad_path=args.jaad_path, data_set='train', balance=False, transforms=transform, forcast=args.forcast)
    te_data = DataSet(path=args.data_path, jaad_path=args.jaad_path, data_set='test', balance=args.balance, transforms=transform, forcast=args.forcast)
    val_data = DataSet(path=args.data_path, jaad_path=args.jaad_path, data_set='val', balance=False, transforms=transform, forcast=args.forcast)
    tr = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(te_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return tr, te, val

def main(args):
    seed_all(args.seed)
    
    tr, te, val = data_loader(args)
    model = LitPedGraph(args, len(tr))
    if not Path(args.logdir).is_dir():
        os.mkdir(args.logdir)
    checkpoint_callback = ModelCheckpoint(
        dirpath = args.logdir,
        monitor = 'val_acc',
        save_top_k = 5,
        filename = 'jaad-{epoch:02d}-{val_acc:.3f}',
        mode = 'max',
        save_weights_only = True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus = [args.device],
        max_epochs = args.epochs,
        auto_lr_find = False,
        callbacks = [checkpoint_callback, lr_monitor],
        precision = 16
    )
    trainer.tune(model, tr)
    trainer.fit(model, tr, val)
    torch.save(model.model.state_dict(), args.logdir + 'last.pth')
    trainer.test(model, te, ckpt_path = 'best')
    torch.save(model.model.state_dict(), args.logdir + 'best.pth')
    print('finish.')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('pedestrian model')
    parser.add_argument('--logdir', type=str, default='./log/JAAD', help='save path')
    parser.add_argument('--device', type=str, default=0, help='choose device.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate to train.')
    parser.add_argument('--data_path', type=str, default='./data/JAAD', help='data path')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the dataloader.')
    parser.add_argument('--forcast', type=bool, default=False, help='Use the human pose forcasting data.')
    parser.add_argument('--jaad_path', type=str, default='./JAAD')
    parser.add_argument('--balance', type=bool, default=True, help='Balnce or not the data set')
    parser.add_argument('--bh', type=str, default='all', help='all or bh, if use all samples or only samples with behaevior labers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch.')

    parser.add_argument('--d_model', type=int, default=256, help='the dimension after embedding.')
    parser.add_argument('--num_layers', type=int, default=4, help='the number of layers.')
    parser.add_argument('--dff', type=int, default=512, help='the number of the units.')
    parser.add_argument('--num_heads', type=int, default=8, help='number of the heads of the multi-head model.')
    parser.add_argument('--encoding_dims', type=int, default=32, help='dimension of the time.')
    parser.add_argument('--bbox_input', type=int, default=4, help='dimension of bbox.')
    parser.add_argument('--vel_input', type=int, default=1, help='dimension of velocity.')
    args = parser.parse_args()
    main(args)
