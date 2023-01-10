import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tools.preprocessing import *
from models.model_jaad import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
import json
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    seed_all(args.seed)
    tr, val, te = data_loader(args)
    train_data = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    valid_data = DataLoader(val, batch_size=args.batch_size, shuffle=True)
    test_data = DataLoader(te, batch_size=8)

    model = Model(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    cls_criterion = nn.BCELoss()
    cross_criterion = nn.MSELoss()

    model_name = 'JAAD'
    checkpoint_filepath = 'checkpoints/JAAD/{}.pt'.format(model_name)
    writer = SummaryWriter('logs/{}'.format(model_name))

    train(model, train_data, valid_data, cls_criterion, cross_criterion, optimizer, checkpoint_filepath, writer, epochs=args.epochs)

    model = Model(args)
    model.to(device)
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    preds, labels = test(model, test_data)
    pred_cpu = torch.Tensor.cpu(preds)
    label_cpu = torch.Tensor.cpu(labels)
    acc = accuracy_score(label_cpu, np.round(pred_cpu))
    f1 = f1_score(label_cpu, np.round(pred_cpu))
    pre_s = precision_score(label_cpu, np.round(pred_cpu))
    recall_s = recall_score(label_cpu, np.round(pred_cpu))
    auc = roc_auc_score(label_cpu, np.round(pred_cpu))
    ave_pre_s = average_precision_score(label_cpu, np.round(pred_cpu))
    conf_matrix = confusion_matrix(label_cpu, np.round(pred_cpu), normalize='true')

    '''results = {}
    results['acc'] = acc
    results['f1'] = f1
    results['precision_score'] = pre_s
    results['recall_score'] = recall_s
    results['roc_auc_score'] = auc
    results['average_precision_score'] = ave_pre_s
    results['confusion_matrix'] = conf_matrix
    config = json.dumps(results)

    f = open('checkpoints/JAAD/{}.json'.format(model_name), 'w')
    f.write(config)
    f.close()'''
    print(f'Acc: {acc}\n f1: {f1}\n precision_score: {pre_s}\n recall_score: {recall_s}\n roc_auc_score: {auc}\n average_precision_score: {ave_pre_s}')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('Pedestrian Model')
    parser.add_argument('--logdir', type=str, default='./log/JAAD', help='save path')
    parser.add_argument('--batch_size', type=int, default=16, help='size of batch.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate to train.')
    parser.add_argument('--data_path', type=str, default='./data/JAAD', help='data path')
    parser.add_argument('--jaad_path', type=str, default='./JAAD')
    parser.add_argument('--bh', type=str, default='all', help='all or bh, if use all samples or only samples with behaevior labers')
    parser.add_argument('--balance', type=bool, default=True, help='balance or not for test dataset.')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--d_model', type=int, default=128, help='the dimension after embedding.')
    parser.add_argument('--num_layers', type=int, default=5, help='the number of layers.')
    parser.add_argument('--dff', type=int, default=256, help='the number of the units.')
    parser.add_argument('--num_heads', type=int, default=8, help='number of the heads of the multi-head model.')
    parser.add_argument('--encoding_dims', type=int, default=32, help='dimension of the time.')
    parser.add_argument('--bbox_input', type=int, default=4, help='dimension of bbox.')
    parser.add_argument('--vel_input', type=int, default=2, help='dimension of velocity.')
    parser.add_argument('--kps_input', type=int, default=4, help='the dimension of keypoints.')
    parser.add_argument('--gcn', type=bool, default=True, help='Use Graph conv networks else use spatial attention.')
    parser.add_argument('--sub', type=int, default=3, help='the number of sub matrix')
    parser.add_argument('--groups', type=int, default=8, help='the groups of the learn matrix.')
    parser.add_argument('--times_num', type=int, default=32, help='')
    args = parser.parse_args()
    main(args)
