import torch
import os
import numpy as np
import random
from torchvision import transforms as A

from tools.jaad_dataloader import DataSet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_all(seed):
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    tr_data = DataSet(path=args.data_path, set_path=args.jaad_path, data_set='train', balance=False, bh=args.bh, transforms=transform)
    te_data = DataSet(path=args.data_path, set_path=args.jaad_path, data_set='test', balance=args.balance, bh=args.bh, transforms=transform)
    val_data = DataSet(path=args.data_path, set_path=args.jaad_path, data_set='val', balance=False, bh=args.bh, transforms=transform)

    return tr_data, val_data, te_data

def binary_acc(label, pred):
    label_tag = torch.round(label)
    correct_results_sum = (label_tag == pred).sum().float()
    acc = correct_results_sum / pred.shape[0]
    return acc

def train(model, train_data, valid_data, cls_criterion, cross_criterion, optimizer, checkpoint_filepath, writer, epochs):
    best_valid_loss = np.inf
    improvement_ratio = 0.01
    best_valid_acc = 0.0
    num_steps_wo_improvement = 0

    for epoch in range(epochs):
        nb_batches_train = len(train_data)
        train_acc = 0
        model.train()
        cls_losses = 0.0
        cross_losses = 0.0
        full_losses = 0.0
        print('Epoch: {} training...'.format(epoch + 1))
        for kps, label, img, bbox, vel, cross_point in train_data:
            kps = kps.to(device)
            label = label.reshape(-1, 1).to(device).float()
            img = img.to(device)
            bbox = bbox.to(device)
            vel = vel.to(device)
            cross_point = cross_point.to(device)

            pred, point, sigma_cls, sigma_cross = model(kps, img, bbox, vel)
            cls_loss = cls_criterion(pred, label)
            cross_loss = cross_criterion(point, cross_point)
            f_loss = cls_loss / (sigma_cls * sigma_cls) + cross_loss / (sigma_cross * sigma_cross) + torch.log(torch.abs(sigma_cls)) + torch.log(torch.abs(sigma_cross))

            model.zero_grad()
            f_loss.backward()

            full_losses += f_loss.item()
            cls_losses += cls_loss.item()
            cross_losses += cross_loss.item()

            optimizer.step()
            
            train_acc += binary_acc(label, torch.round(pred))

        writer.add_scalar('training Full loss', full_losses / nb_batches_train, epoch + 1)
        writer.add_scalar('training Class loss', cls_losses / nb_batches_train, epoch + 1)
        writer.add_scalar('training Cross_point loss', cross_losses / nb_batches_train, epoch + 1)
        writer.add_scalar('training Acc', train_acc / nb_batches_train, epoch + 1)
        print('Sigma_cls: ' + str(sigma_cls) + '  Sigma_crossp: ' + str(sigma_cross))
        print(f'Epoch {epoch + 1}: | Train_Full_Loss {full_losses / nb_batches_train} | Train_Class_Loss {cls_losses / nb_batches_train} | Train_Crossp_Loss {cross_losses / nb_batches_train} | Train_Acc {train_acc / nb_batches_train} ')

        valid_full_loss, valid_cls_loss, valid_cross_loss, val_acc = evaluate(model, valid_data, cls_criterion, cross_criterion)

        writer.add_scalar('validation Full loss', valid_full_loss / nb_batches_train, epoch + 1)
        writer.add_scalar('validation Class loss', valid_cls_loss / nb_batches_train, epoch + 1)
        writer.add_scalar('validation Cross_point loss', valid_cross_loss / nb_batches_train, epoch + 1)
        writer.add_scalar('validation Acc', val_acc / nb_batches_train, epoch + 1)
        
        if num_steps_wo_improvement == 10:
            print('Early stopping on epoch:{}'.format(str(epoch)))
            break
        if val_acc >= best_valid_acc or (best_valid_loss - valid_cls_loss) > np.abs(best_valid_loss * improvement_ratio):
            num_steps_wo_improvement = 0
            print('Update improvement.\n')
            best_valid_acc = val_acc
            best_valid_loss = valid_cls_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'LOSS': full_losses / nb_batches_train,}, checkpoint_filepath)
        else:
            num_steps_wo_improvement += 1
            print('Not update.\n')


def evaluate(model, val_data, cls_criterion, cross_criterion):
    nb_batches = len(val_data)
    val_full_losses = 0.0
    val_cls_losses = 0.0
    val_cross_losses = 0.0
    print('in Validation...')
    with torch.no_grad():
        model.eval()
        acc = 0
        for kps, label, img, bbox, vel, cross_point in val_data:
            kps = kps.to(device)
            label = label.reshape(-1, 1).to(device).float()
            img = img.to(device)
            bbox = bbox.to(device)
            vel = vel.to(device)
            cross_point = cross_point.to(device)

            pred, point, sigma_cls, sigma_cross = model(kps, img, bbox, vel)
            val_cls_loss = cls_criterion(pred, label)
            val_cross_loss = cross_criterion(point, cross_point)
            val_f_loss = val_cls_loss / (sigma_cls * sigma_cls) + val_cross_loss / (sigma_cross * sigma_cross) + torch.log(torch.abs(sigma_cls)) + torch.log(torch.abs(sigma_cross))

            val_full_losses += val_f_loss.item()
            val_cls_losses += val_cls_loss.item()
            val_cross_losses += val_cross_loss.item()

            acc += binary_acc(label, torch.round(pred))
    print('Sigma_cls: ' + str(sigma_cls) + '  Sigma_crossp: ' + str(sigma_cross))
    print(f'Valid_Full_Loss {val_full_losses / nb_batches} | Valid_Cls_Loss {val_cls_losses / nb_batches} | Valid_Crossp_Loss {val_cross_losses / nb_batches} | Valid_Acc {acc / nb_batches} \n')
    return val_full_losses / nb_batches, val_cls_losses / nb_batches, val_cross_losses / nb_batches, acc / nb_batches


def test(model, test_data):
    print('Tesing...')
    with torch.no_grad():
        model.eval()
        step = 0
        for kps, label, img, bbox, vel, cross_point in test_data:
            kps = kps.to(device)
            label = label.reshape(-1, 1).to(device).float()
            img = img.to(device)
            bbox = bbox.to(device)
            vel = vel.to(device)
            cross_point = cross_point.to(device)

            pred, point, sigma_cls, sigma_cross = model(kps, img, bbox, vel)
            if step == 0:
                preds = pred
                labels = label
            else:
                preds = torch.cat((preds, pred), 0)
                labels = torch.cat((labels, label), 0)
            step += 1

    return preds, labels

