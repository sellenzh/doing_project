import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from pytorch_lightning.callbacks import plc

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args


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
    if args.la_scheduler:
        callbacks.append(plc.LearningrateMonitor(
            logging_unterval='step'#epoch
        ))
    return callbacks

def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()

    #Basic Trianing Control
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.002)
    #LR scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_min_lr', type=float, default=1e-5)
    #Restart Contorl
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)
    # Training Info
    parser.add_argument('--dataset', default='standard_data', type=str)
    parser.add_argument('--data_dir', default='./data/PIE', type=str)
    parser.add_argument('--model_name', default='PedModel', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='./pie-train/', type=str)
    # Model Hyper_parameters
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)
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
