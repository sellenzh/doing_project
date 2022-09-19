
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse
from tqdm import tqdm

from torchvision import transforms as A
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

from final_pie_dataloder import DataSet
from model0906.ped_graph import PedModel


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def data_loader(args):
    transform = A.Compose(
        [
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


class StatsClass:
    def __init__(self, pcpa_pred, pedmodel_pred, y) -> None:
        self.pcpa = pcpa_pred
        self.pedmodel_pred = pedmodel_pred
        self.onehot_y = np.eye(3)[y.reshape(-1).astype(np.int32)]
        y[y == 2] == 0
        self.y = y
        self.results = {accuracy_score.__name__: [0.0, 0.0, ], }
        self.models_k = ['PCPA', 'PedModel+']
        self.results = pd.DataFrame(self.results, index=self.models_k)

    def stats(self):
        pcpa = np.round(self.pcpa)
        pcpa_rn = self.pcpa.astype(np.int64)
        pedmodel_pred = np.round(self.pedmodel_pred)
        pedmodel_pred_rn = self.pedmodel_pred.astype(np.int64)

        # stats_fn.stats(accuracy_score, True, 100)
        self.results.at[self.models_k[0], accuracy_score.__name__] = accuracy_score(self.y, pcpa_rn, normalize=True,
                                                                                    sample_weight=None) * 100
        self.results.at[self.models_k[1], accuracy_score.__name__] = accuracy_score(self.y, pedmodel_pred_rn,
                                                                                    normalize=True,
                                                                                    sample_weight=None) * 100

        # stats_fn.stats(balanced_accuracy_score, True, 100)
        self.results.at[self.models_k[0], balanced_accuracy_score.__name__] = balanced_accuracy_score(self.y, pcpa_rn,
                                                                                                      sample_weight=None,
                                                                                                      adjusted=False) * 100
        self.results.at[self.models_k[1], balanced_accuracy_score.__name__] = balanced_accuracy_score(self.y,
                                                                                                      pedmodel_pred_rn,
                                                                                                      sample_weight=None,
                                                                                                      adjusted=False) * 100

        # stats_fn.stats(f1_score, True)
        self.results.at[self.models_k[0], f1_score.__name__] = f1_score(self.y, pcpa_rn, average=None,
                                                                        sample_weight=None)
        self.results.at[self.models_k[1], f1_score.__name__] = f1_score(self.y, pedmodel_pred_rn, average=None,
                                                                        sample_weight=None)

        # stats_fn.stats(precision_score, True)
        self.results.at[self.models_k[0], precision_score.__name__] = precision_score(self.y, pcpa_rn, average=None,
                                                                                      sample_weight=None)
        self.results.at[self.models_k[1], precision_score.__name__] = precision_score(self.y, pedmodel_pred_rn,
                                                                                      average=None, sample_weight=None)

        # stats_fn.stats(recall_score, True)
        self.results.at[self.models_k[0], recall_score.__name__] = recall_score(self.y, pcpa_rn, average=None,
                                                                                sample_weight=None)
        self.results.at[self.models_k[1], recall_score.__name__] = recall_score(self.y, pedmodel_pred_rn, average=None,
                                                                                sample_weight=None)

        # stats_fn.stats(roc_auc_score, False)
        self.results.at[self.models_k[0], roc_auc_score.__name__] = roc_auc_score(self.y, pcpa, average='macro',
                                                                                  sample_weight=None)
        self.results.at[self.models_k[1], roc_auc_score.__name__] = roc_auc_score(self.y, pedmodel_pred,
                                                                                  average='macro', sample_weight=None)

        # stats_fn.stats(average_precision_score, False)
        self.results.at[self.models_k[0], average_precision_score.__name__] = average_precision_score(self.y, pcpa,
                                                                                                      average='macro',
                                                                                                      sample_weight=None)
        self.results.at[self.models_k[1], average_precision_score.__name__] = average_precision_score(self.y,
                                                                                                      pedmodel_pred,
                                                                                                      average='macro',
                                                                                                      sample_weight=None)


class MusterModel(nn.Module):
    def __init__(self, args):
        super(MusterModel, self).__init__()

        self.model = PedModel(n_clss=3)
        pth = torch.load(args.pth, map_location=args.device)
        self.model.load_state_dict(pth)
        self.model = self.model.to(args.device)
        self.model.eval()

    def forward(self, pose, frame, vel):
        with torch.no_grad():
            out = self.model(pose, frame, vel).softmax(1)
        return out


def test(model, data, args):
    str_t = torch.cuda.Event(enable_timing=True)
    end_t = torch.cuda.Event(enable_timing=True)
    timing = []

    ys = np.zeros([len(data), 1])
    pedmodel_pred_all = np.zeros([len(data), 3])
    pedmodel_pred = np.zeros([len(data), 1])
    pcpa_pred = np.zeros([len(data), 1])

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(data, desc='Testing samples')):
            pose, y, frame, vel = batch[0].float().to(args.device), batch[1].long().to(args.device), batch[2].float().to(args.device), batch[3].float().to(args.device)
            models_data = batch[-1]

            str_t.record()
            if args.last2:
                pose = pose[:, :, -(2 + 30):] if args.forecast else pose[:, :, -2:]
                pred = model(pose.contiguous().half(), frame.half(), vel.half())
            else:
                pred = model(pose.half(), frame.half(), vel.half())
            end_t.record()
            torch.cuda.synchronize()
            timing.append(str_t.elapsed_time(end_t))

            ys[i] = int(y.item())
            pedmodel_pred_all[i] = pred.detach().cpu().numpy()

            if args.argmax:
                prov = pred[:, pred.argmax(1)].cpu().numpy()
                prov = 1 - prov if pred.argmax(1) != 1 else prov
                pedmodel_pred[i] = prov
            else:
                pred[:, 0] = min(1, pred[:, 0] + pred[:, 2])
                pred[:, 1] = max(0, pred[:, 1] - pred[:, 2])
                pedmodel_pred[i] = pred[:, 1].item() if pred.argmax(1) == 1 else 1 - pred[:, 0].item()

            pcpa_pred[i] = models_data[0].cpu().numpy()

            y[y==2] = 0
            assert y.item() == models_data[1].item(), 'labels sanity check'
    y = ys.copy()
    y[y == 2] = 0
    pedmodel_pred = np.clip(pedmodel_pred, 0, 1)

    return y, pcpa_pred, pedmodel_pred, ys, pedmodel_pred_all, timing


def main(args):
    seed_all(args.seed)

    data = data_loader(args)
    model = MusterModel(args)
    model = model.to(args.device)

    model_folder_name = 'PedModel'
    checkpoint_filepath = "checkpoints/{}.pth".format(model_folder_name)
    checkpoint = torch.load(checkpoint_filepath, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    y, pcpa_pred, pedmodel_pred, ys, pedmodel_pred_all, timing = test(model, data, args)

    stats_fn = StatsClass(pcpa_pred, pedmodel_pred, ys)

    stats_fn.stats()

    print(f'balance data: {args.balance}, bh: {args.bh}, last2: {args.last2}, Model: ' + args.ckpt.split('/')[-2])
    print(stats_fn.results)
    print(np.mean((pedmodel_pred_all[:, 0] > 0.5) == stats_fn.onehot_y[:, 0]))
    print(np.mean((pedmodel_pred_all[:, 1] > 0.5) == stats_fn.onehot_y[:, 1]))
    print(np.mean((pedmodel_pred_all[:, 2] > 0.5) == stats_fn.onehot_y[:, 2]))

    print(*['-'] * 30)
    print(f'Average run time of Pedestrian Graph +: {np.mean(timing):.3f}')
    print('finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pedestrian prediction crossing")
    parser.add_argument('--pth', type=str, default="./checkpoints/PedModel.pth", help="Path to model weigths")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU")
    parser.add_argument('--data_path', type=str, default='./data/PIE', help='Path to the train and test data')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training and test")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the dataloader")
    parser.add_argument('--frames', type=bool, default=True, help='activate the use of raw frames')
    parser.add_argument('--velocity', type=bool, default=True, help='activate the use of the odb and gps velocity')
    parser.add_argument('--seg', type=bool, default=True, help='Use the segmentation map')
    parser.add_argument('--H3D', type=bool, default=True, help='Use 3D human keypoints')
    parser.add_argument('--forcast', type=bool, default=True, help='Use the human pose forcasting data')
    parser.add_argument('--pie_path', type=str, default='./PIE')
    parser.add_argument('--bh', type=str, default='all',
                        help='all or bh, if use all samples or only samples with behaevior labers')
    parser.add_argument('--balance', type=bool, default=True, help='Balance or not the data set')
    parser.add_argument('--pcpa', type=str, default='./data', help='path with results for pcpa')
    parser.add_argument('--last2', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--argmax', type=bool, default=True, help='Use argemax funtion, if false use maping')
    parser.add_argument('--forecast', type=bool, default=True, help='Use the human pose forcasting data')
    args = parser.parse_args()
    main(args)
