import torch
from torchvision import transforms as A
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from pie_dataloader23 import DataSet
from model0906.ped_graph import PedModel
from pathlib import Path
import argparse
import os
import numpy as np
import random
import json

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


def train_criterion_weight():
    tr_samples = [9974, 5956, 7876]
    tr_weight = torch.from_numpy(np.min(tr_samples) / tr_samples).float().cuda()
    return tr_weight

def test_criterion_weight():
    te_samples = [9921, 5346, 3700]
    te_weight = torch.from_numpy(np.min(te_samples) / te_samples).float().cuda()
    return te_weight

def valid_criterion_weight():
    val_samples = [3404, 1369, 1813]
    val_weight = torch.from_numpy(np.min(val_samples) / val_samples).float().cuda()
    return val_weight

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]

    return acc


def train(model, tr, val, optimizer, checkpoint_filepath, writer, args):
    best_valid_loss = np.inf
    improvement_ratio = 0.01
    num_steps_wo_improvement = 0

    for epoch in range(args.epochs):
        nb_batches_train = len(tr)
        train_acc = 0
        model.train()
        losses = 0.0

        for pose, y, frame, vel in tr:
            # TO CUDA
            pose, y, frame, vel = pose.to(args.device), y.to(args.device), frame.to(args.device), vel.to(args.device)
            # preprocess pose data
            if np.random.randint(10) >= 5 and args.time_crop:
                crop_size = np.random.randint(2, 21)
                pose = pose[:, :, -crop_size:]

            out = model(pose, frame, vel)
            w = None if args.balance else train_criterion_weight()
            y_onehot = torch.FloatTensor(y.shape[0], 3).to(y.device).zero_()
            y_onehot.scatter_(1, y.long(), 1)

            loss = F.binary_cross_entropy_with_logits(out, y_onehot, weight=w)

            model.zero_grad()
            #optimizer
            optimizer.zero_grad()
            loss.backward()
            losses += loss.item()
            optimizer.step()
            pred = out.softmax(1).argmax(1)
            acc = balanced_accuracy_score(pred.view(-1).long().cpu(), y.view(-1).long().cpu())
            train_acc += acc

        writer.add_scalar('training loss', losses / nb_batches_train, epoch + 1)
        writer.add_scalar('training Acc', train_acc / nb_batches_train, epoch + 1)

        print(f"Epoch {epoch}: | Train_Loss {losses / nb_batches_train} | Train_Acc {train_acc / nb_batches_train} ")
        valid_loss, val_acc = evaluate(model, val, args)

        writer.add_scalar('validation loss', valid_loss, epoch + 1)
        writer.add_scalar('validation Acc', val_acc, epoch + 1)

        if (best_valid_loss - valid_loss) > np.abs(best_valid_loss * improvement_ratio):
            num_steps_wo_improvement = 0
        else:
            num_steps_wo_improvement += 1

        if num_steps_wo_improvement == 10:
            print("Early stooping on epoch:{}".format(str(epoch)))
            break
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("Save best model on epoch:{}".format(str(epoch)))
            torch.save(model.state_dict(), checkpoint_filepath)
            '''torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'LOSS': losses / nb_batches_train,
            }, checkpoint_filepath)'''


def evaluate(model, data_loader, args):
    nb_batches = len(data_loader)
    val_losses = 0.0
    with torch.no_grad():
        model.eval()
        val_acc = 0
        for pose, y, frame, vel in data_loader:
            #TO CUDA
            pose, y, frame, vel = pose.to(args.device), y.to(args.device), frame.to(args.device), vel.to(args.device)
            #Prediction
            out = model(pose, frame, vel)
            w = None if args.balance else valid_criterion_weight()#balance the data

            y_onehot = torch.FloatTensor(y.shape[0], 3).to(y.device).zero_()
            y_onehot.scatter_(1, y.long(), 1)

            val_loss = F.binary_cross_entropy_with_logits(out, y_onehot, weight=w)
            val_losses += val_loss.item()

            pred = out.softmax(1).argmax(1)
            acc = balanced_accuracy_score(pred.view(-1).long().cpu(), y.view(-1).long().cpu())
            val_acc += acc

    print(f"Validation_Loss {val_losses / nb_batches} | Val_Acc {val_acc / nb_batches} \n")
    return val_losses / nb_batches, val_acc / nb_batches


def test(model, data_loader, args):
    with torch.no_grad():
        model.eval()
        step = 0
        for pose, y, frame, vel in data_loader:
            pose = pose.to(args.device)
            y = y.to(args.device)
            frame = frame.to(args.device)
            vel = vel.to(args.device)

            out = model(pose, frame, vel)
            if (step == 0):
                pred = out
                labels = y

            else:
                pred = torch.cat((pred, out), 0)
                labels = torch.cat((labels, y), 0)
            step += 1

    return pred, labels


def main(args):
    seed_all(args.seed)

    args.frames = True
    args.velocity = True
    args.seg = True
    args.forecast = True
    args.time_crop = True
    args.H3D = True


    tr, te, val = data_loader(args)
    model = PedModel(n_clss=3)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, div_factor=10.0, total_steps=args.lr * len(tr), verbose=False)

    model_folder_name = 'PedModel'
    checkpoint_filepath = "checkpoints/{}.pth".format(model_folder_name)
    writer = SummaryWriter('logs/{}'.format(model_folder_name))

    train(model, tr, val, optimizer, checkpoint_filepath, writer, args)

    '''checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    pred, lab = test(model, te, args)
    pred_cpu = torch.Tensor.cpu(pred)
    lab_cpu = torch.Tensor.cpu(lab)
    acc = accuracy_score(lab_cpu, np.round(pred_cpu)).argmax(axis=1)
    conf_matrix = confusion_matrix(lab_cpu, np.round(pred_cpu), normalize='true')
    f1 = f1_score(lab_cpu, np.round(pred_cpu))
    auc = roc_auc_score(lab_cpu, np.round(pred_cpu))
    #torch.save(model.state_dict(), args.logdir + 'best.pth')

    print(f"Accuracy: {acc} \n AUC: {auc} \n f1: {f1}")'''


def save_model(name, model, args):
    path = args.logdir
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path + '/epoch' + name + '.pth')



if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser("Pedestrian prediction crossing")
    parser.add_argument('--logdir', type=str, default="./pie-23-IVSFT/",help="logger directory for tensorboard")
    parser.add_argument('--device', type=str, default=0, help="GPU")
    parser.add_argument('--epochs', type=int, default=200, help="Number of eposch to train")
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate to train')
    parser.add_argument('--data_path', type=str, default='./data/PIE', help='Path to the train and test data')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and test")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the dataloader")
    parser.add_argument('--frames', type=bool, default=True, help='avtivate the use of raw frames')
    parser.add_argument('--velocity', type=bool, default=True, help='activate the use of the odb and gps velocity')
    parser.add_argument('--seg', type=bool, default=True, help='Use the segmentation map')
    parser.add_argument('--forecast', type=bool, default=True, help='Use the human pose forcasting data')
    parser.add_argument('--time_crop', type=bool, default=True, help='Use random time trimming')
    parser.add_argument('--H3D', type=bool, default=True, help='Use 3D human keypoints')
    parser.add_argument('--pie_path', type=str, default='./PIE')
    parser.add_argument('--balance', type=bool, default=True, help='Balnce or not the data set')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
