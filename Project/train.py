import torch
from torchvision import transforms as A
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import numpy as np
import random
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from pytorch_lightning.callbacks import plc

from model.model_interface import MInterface
#from data.data_interface import DInterface
from pie_dataloader23 import DataSet

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def data_loader(args):
    transform = A.Compose(
        [
            A.ToPILImage(),
            A.RandomPosterize(bits=2),
            A.RandomInvert(p=0.2),
            A.RandomSolarize(threshold=50.0),
            A.RandomAdjustSharpness(sharpness_factor=2),
            A.RandomAutocontrast(p=0.2),
            A.RandomEqualize(p=0.2),
            A.ColorJitter(0.5, 0.3),
            A.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    tr_data = DataSet(path=args.data_path, pie_path=args.pie_path, data_set='train', frame=True, vel=True,
                      balance=False, transforms=transform, seg_map=args.seg, h3d=args.H3D, forecast=args.forecast)
    te_data = DataSet(path=args.data_path, pie_path=args.pie_path, data_set='test', frame=True, vel=True, balance=False,
                      transforms=transform, seg_map=args.seg, h3d=args.H3D, t23=False, forecast=args.forecast)
    val_data = DataSet(path=args.data_path, pie_path=args.pie_path, data_set='val', frame=True, vel=True, balance=False,
                       transforms=transform, seg_map=args.seg, h3d=args.H3D, forecast=args.forecast)

    tr = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    worker_init_fn=worker_init_fn, pin_memory=True)
    te = DataLoader(te_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    worker_init_fn=worker_init_fn, pin_memory=True)
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                     worker_init_fn=worker_init_fn, pin_memory=True)

    return tr, te, val

def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))
    callbacks.append(plc.ModelCheckpoint(
        #dirpath=args.logdir,
        monitor='val_acc',
        filename='pie-{epoch:02d}-{val_acc:.3f}',
        save_top_k=5,
        mode='max',
        save_last=True
    ))
    callbacks.append(plc.LearningrateMonitor(
            logging_unterval='step'))
    return callbacks

def main(args):
    seed_everything(args.seed)
    pl.seed_everything(args.seed)

    tr, te, val = data_loader(args)

    model = MInterface(args, len(tr))
    if not Path(args.logdir).is_dir():
        os.mkdir(args.logdir)
    callbacks = load_callbacks()
    trainer = pl.Trainer(
        gpus=[args.device], max_epochs=args.epochs,
        auto_lr_find=False, callbacks=callbacks[-2:], precision=16,
    )
    trainer.tune(model, tr)
    trainer.fit(model, tr, val)
    trainer.save(model.model.state_dict(), args.logdir + 'last.pth')
    trainer.test(model, te, ckpt_path='best')
    trainer.save(model.model.state_dict(), args.logdir + 'best.pth')
    print('finish!')



if __name__ == '__main__':
    parser = ArgumentParser()
    torch.cuda.empty_cache()


    #Basic Trianing Control
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.002)
    #LR scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_min_lr', type=float, default=1e-5)
    '''#Restart Contorl
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)'''
    # Training Info
    parser.add_argument('--dataset', default='standard_data', type=str)
    parser.add_argument('--data_dir', default='./data/PIE', type=str)
    parser.add_argument('--data_path', default='./data/PIE', type=str)
    parser.add_argument('--pie_path', default='./PIE', type=str)
    parser.add_argument('--balance', type=bool, default=True)
    parser.add_argument('--model_name', default='PedModel', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='./pie-23-IVSFT/', type=str)
    # Model Hyper_parameters
    parser.add_argument('--frames', type=bool, default=True)
    parser.add_argument('--velocity', type=bool, default=True)
    parser.add_argument('--seg', type=bool, default=True)
    parser.add_argument('--forecast', type=bool, default=False)
    parser.add_argument('--time_crop', type=bool, default=True)
    parser.add_argument('--H3D', type=bool, default=True)
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--head_num', default=8, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)
    parser.add_argument('--nodes', default=19, type=int)
    parser.add_argument('--groups', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    #parser.add_argument('--time_attention_layers_num', default=5, type=int)
    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)
    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)
    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)
    args = parser.parse_args()
    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]
    main(args)