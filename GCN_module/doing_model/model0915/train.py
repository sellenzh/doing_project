import torch
from torchvision import transforms as A
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from skleaern.metrics import balanced_accuracy_score

from pie_dataloader23 import DataSet
from model.ped_graph import PedGraph
from pathlib import Path
import argparse
import os
import numpy as np
import random

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


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


def train(model, tr, val, optimizer, checkpoint_filepath, writer, args):
    best_valid_loss = np.inf
    improvement_ratio = 0.01
    num_steps_wo_improvement = 0

    for epoch in args.epochs:
        nb_batches_train = len(tr)
        train_acc = 0
        model.train()
        losses = 0.0

        for pose, y, frame, vel in tr:
            pose = pose.to(args.device)


def main(args):
    seed_all(args.seed)

    args.frames = True
    args.velocity = True
    args.seg = True
    args.forecast = True
    args.time_crop = True
    args.H3D = True

    tr, te, val = data_loader(args)
    model = PedGraph(n_clss=3)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameter(), lr=args.lr, weight_decay=1e-3)

    model_folder_name = 'PedModel'
    checkpoint_filepath = "checkpoints/{}.pt".format(model_folder_name)
    writer = SummaryWriter('logs/{}'.format(model_folder_name))

    train(model, tr, val, optimizer, checkpoint_filepath, writer, args)







if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser("Pedestrian prediction crossing")
    parser.add_argument('--logdir', type=str, default="./test0810/pie-23-IVSFT/",help="logger directory for tensorboard")
    parser.add_argument('--device', type=str, default=0, help="GPU")
    parser.add_argument('--epochs', type=int, default=30, help="Number of eposch to train")
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate to train')
    parser.add_argument('--data_path', type=str, default='./data/PIE', help='Path to the train and test data')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and test")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the dataloader")
    parser.add_argument('--frames', type=bool, default=False, help='avtivate the use of raw frames')
    parser.add_argument('--velocity', type=bool, default=False, help='activate the use of the odb and gps velocity')
    parser.add_argument('--seg', type=bool, default=False, help='Use the segmentation map')
    parser.add_argument('--forecast', type=bool, default=False, help='Use the human pose forcasting data')
    parser.add_argument('--time_crop', type=bool, default=False, help='Use random time trimming')
    parser.add_argument('--H3D', type=bool, default=True, help='Use 3D human keypoints')
    parser.add_argument('--pie_path', type=str, default='./PIE')
    parser.add_argument('--balance', type=bool, default=True, help='Balnce or not the data set')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
