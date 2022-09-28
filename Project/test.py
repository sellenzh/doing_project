import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms as A
from finale_pie_dataloader import DataSet
from model.standard_net import PedModel
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def data_loader(args):
    transform = A.Compose(
        [A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    te_data = DataSet(
        path=args.data_path,
        pie_path=args.pie_path,
        frame=True,
        vel=True,
        seg_map=args.seg,
        h3d=args.H3D,
        balance=args.balance,
        bh=args.bh,
        t23=args.balance,
        transforms=transform,
        pcpa=args.pcpa,
        forecast=args.forecast,
        last2=args.last2
    )
    te = DataLoader(te_data, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True)

    return te

class TestModule:
    def __init__(self, pcpa_pred, pedgraph_pred, y):
        self.pcpa = pcpa_pred
        self.pedgraph_pred = pedgraph_pred
        self.onehot_y = np.eye(3)[y.reshape(-1).astype(np.int32)]
        y[y == 2] = 0
        self.y = y
        self.results = {
            accuracy_score.__name__: [0.0, 0.0, ],
        }
        self.models_k = ['PCPA', 'PedGraph+']
        self.results = pd.DataFrame(self.results, index=self.models_k)

    def stats(self, fn, rn, mult=1):
        pcpa = np.round(self.pcpa) if rn else self.pcpa
        pedgraph_pred = np.round(self.pedgraph_pred) if rn else self.pedgraph_pred

        self.results.at[self.models_k[0], fn.__name__] = fn(self.y, pcpa) * mult
        self.results.at[self.models_k[1], fn.__name__] = fn(self.y, pedgraph_pred) * mult


class MusterModel(nn.Module):
    def __init__(self, args):
        super(MusterModel, self).__init__()

        self.model = PedModel(args, n_clss=3)
        ckpt = torch.load(args.ckpt, map_location=args.device)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(args.device)
        self.model.eval()

    def forward(self, x, f, v):
        with torch.no_grad():
            cx = self.model(x, f, v).softmax(1)
            # cx = self.model(x, f, v).sigmoid()
        return cx


def main(args):
    seed_everything(args.seed)

    data = data_loader(args)
    model = MusterModel(args).eval()
    model.half()
    str_t = torch.cuda.Event(enable_timing=True)
    end_t = torch.cuda.Event(enable_timing=True)
    timimg = []

    ys = np.zeros([len(data), 1])

    pedgraph_pred_all = np.zeros([len(data), 3])
    pedgraph_pred = np.zeros([len(data), 1])
    pcpa_pred = np.zeros([len(data), 1])

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data, desc='Testing samples')):

            x = batch[0].float().to(args.device)
            y = batch[1].long().to(args.device)
            f = batch[2].float().to(args.device) if args.frames else None
            v = batch[3].float().to(args.device) if args.velocity else None
            models_data = batch[-1]

            str_t.record()
            if args.last2:
                x = x[:, :, -(2 + 30):] if args.forecast else x[:, :, -2:]
                pred = model(x.contiguous().half(), f.half(), v.half())
            else:
                pred = model(x.half(), f.half(), v.half())
            end_t.record()

            torch.cuda.synchronize()
            timimg.append(str_t.elapsed_time(end_t))

            ys[i] = int(y.item())
            pedgraph_pred_all[i] = pred.detach().cpu().numpy()

            if args.argmax:
                prov = pred[:, pred.argmax(1)].cpu().numpy()
                prov = 1 - prov if pred.argmax(1) != 1 else prov
                pedgraph_pred[i] = prov
            else:
                pred[:, 0] = min(1, pred[:, 0] + pred[:, 2])
                pred[:, 1] = max(0, pred[:, 1] - pred[:, 2])
                pedgraph_pred[i] = pred[:, 1].item() if pred.argmax(1) == 1 else 1 - pred[:, 0].item()

            pcpa_pred[i] = models_data[0].cpu().numpy()

            y[y == 2] = 0
            assert y.item() == models_data[1].item(), 'labels sanity check'

    y = ys.copy()
    y[y == 2] = 0
    pedgraph_pred = np.clip(pedgraph_pred, 0, 1)
    stats_fn = TestModule(pcpa_pred, pedgraph_pred, ys)

    stats_fn.stats(accuracy_score, True, 100)
    stats_fn.stats(balanced_accuracy_score, True, 100)
    stats_fn.stats(f1_score, True)
    stats_fn.stats(precision_score, True)
    stats_fn.stats(recall_score, True)
    stats_fn.stats(roc_auc_score, False)
    stats_fn.stats(average_precision_score, False)

    print(f'balance data: {args.balance}, bh: {args.bh}, last2: {args.last2}, Model: ' + args.ckpt.split('/')[-2])
    print(stats_fn.results)
    print(np.mean((pedgraph_pred_all[:, 0] > 0.5) == stats_fn.one_hot_y[:, 0]))
    print(np.mean((pedgraph_pred_all[:, 1] > 0.5) == stats_fn.one_hot_y[:, 1]))
    print(np.mean((pedgraph_pred_all[:, 2] > 0.5) == stats_fn.one_hot_y[:, 2]))

    print(*['-'] * 30)
    print(f'Average frun time of Pedestrian Graph +: {np.mean(timimg):.3f}')
    print('finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pedestrian prediction crosing")
    parser.add_argument('--ckpt', type=str, default="./pie-23-IVSFT/best.pth", help="Path to model weigths")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU")
    parser.add_argument('--data_path', type=str, default='./data/PIE', help='Path to the train and test data')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training and test")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the dataloader")
    parser.add_argument('--frames', type=bool, default=False, help='avtivate the use of raw frames')
    parser.add_argument('--velocity', type=bool, default=False, help='activate the use of the odb and gps velocity')
    parser.add_argument('--seg', type=bool, default=False, help='Use the segmentation map')
    parser.add_argument('--H3D', type=bool, default=True, help='Use 3D human keypoints')
    parser.add_argument('--forcast', type=bool, default=False, help='Use the human pose forcasting data')
    parser.add_argument('--pie_path', type=str, default='./PIE')
    parser.add_argument('--bh', type=str, default='all',
                        help='all or bh, if use all samples or only samples with behaevior labers')
    parser.add_argument('--balance', type=bool, default=False, help='Balnce or not the data set')
    parser.add_argument('--pcpa', type=str, default='./data', help='path with results for pcpa')
    parser.add_argument('--last2', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--argmax', type=bool, default=False, help='Use argemax funtion, if false use maping')
    args = parser.parse_args()
    main(args)


