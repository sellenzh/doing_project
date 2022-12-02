import numpy as np
from tools.pie_data import PIE
from tools.preprocessing import *
from tools.ted_train_utils import train, test

from model.model import Model

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_opts = {
    'fstride': 1,
    'sample_type': 'all',
    'height_rng': [0, float('inf')],
    'squarify_ratio': 0,
    'data_split_type': 'random', #kfold, random, default
    'seq_type': 'crossing', #crossing, intention, all
    'min_track_size': 15,
    'kflod_param': {'num_folds': 1, 'fold': 1},
    'random_params': {'ratios': [0.7, 0.15, 0.15], 'val_data': True, 'regen_data': False},
    'tte': [30, 60],
    'batch_size': 16
}

args = {
    'logdir': "./PIE_dataset",
    'seed': 42,
    'model_name': 'bbox_vel2intention',
    'tte': data_opts['tte'],
    'num_layers': 4,
    'd_model': 256,
    'dff': 512,
    'bbox_input': 4,
    'vel_input': 2,
    'num_heads': 8,
    'encoding_dims': 16,
    'batch_size': 64,
    #'warmup': 1000,
    'epochs': 200,
    'lr': 1e-4
}

def change_type(byte):
    if isinstance(byte, bytes):
        return str(byte, encoding="utf-8")
    return json.JSONEncoder.default(byte)

def pre_data(args):
    imdb = PIE(data_path=args['logdir'])
    seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    balanced_seq_train = balance_dataset(seq_train)
    obs_seq_train, traj_seq_train = tte_dataset(balanced_seq_train, data_opts['tte'], 0.8, 16)


    seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
    balanced_seq_valid = balance_dataset(seq_valid)
    obs_seq_valid, traj_seq_valid = tte_dataset(balanced_seq_valid,  data_opts['tte'], 0, 16)


    seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    obs_seq_test, traj_seq_test = tte_dataset(seq_test, data_opts['tte'], 0, 16)


    bbox_enc_train = obs_seq_train['bbox']
    bbox_enc_valid = obs_seq_valid['bbox']
    bbox_enc_test  = obs_seq_test['bbox']


    bbox_dec_train = traj_seq_train['bbox']
    bbox_dec_valid = traj_seq_valid['bbox']
    bbox_dec_test  = traj_seq_test['bbox']

    action_train = obs_seq_train['activities']
    action_valid = obs_seq_valid['activities']
    action_test  = obs_seq_test['activities']

    vel_train_obd = torch.Tensor(np.array(obs_seq_train['obd_speed']))
    vel_train_gps = torch.Tensor(np.array(obs_seq_train['gps_speed']))
    vel_valid_obd = torch.Tensor(np.array(obs_seq_valid['obd_speed']))
    vel_valid_gps = torch.Tensor(np.array(obs_seq_valid['gps_speed']))
    vel_test_obd = torch.Tensor(np.array(obs_seq_test['obd_speed']))
    vel_test_gps = torch.Tensor(np.array(obs_seq_test['gps_speed']))

    vel_train = torch.cat((vel_train_obd, vel_train_gps), dim=-1)
    vel_valid = torch.cat((vel_valid_obd, vel_valid_gps), dim=-1)
    vel_test = torch.cat((vel_test_obd, vel_test_gps), dim=-1)


    normalized_bbox_enc_train = normalize_bbox(bbox_enc_train)
    normalized_bbox_enc_valid = normalize_bbox(bbox_enc_valid)
    normalized_bbox_enc_test  = normalize_bbox(bbox_enc_test)

    normalized_bbox_dec_train = normalize_bbox(bbox_dec_train)
    normalized_bbox_dec_valid = normalize_bbox(bbox_dec_valid)
    normalized_bbox_dec_test  = normalize_bbox(bbox_dec_test)

    label_action_train = prepare_label(action_train)
    label_action_valid = prepare_label(action_valid)
    label_action_test = prepare_label(action_test)

    X_train_enc, X_valid_enc = torch.Tensor(normalized_bbox_enc_train), torch.Tensor(normalized_bbox_enc_valid)
    Y_train, Y_valid = torch.Tensor(label_action_train), torch.Tensor(label_action_valid)
    X_test_enc, Y_test = torch.Tensor(normalized_bbox_enc_test), torch.Tensor(label_action_test)

    X_train_enc = torch.cat((X_train_enc, vel_train), dim=-1)#->[..., 16, 6]
    X_valid_enc = torch.cat((X_valid_enc, vel_valid), dim=-1)
    X_test_enc = torch.cat((X_test_enc, vel_test), dim=-1)

    X_train_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_train, 60))
    X_valid_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_valid, 60))
    X_test_dec = torch.Tensor(pad_sequence(normalized_bbox_dec_test, 60))

    trainset = TensorDataset(X_train_enc, X_train_dec, Y_train)
    validset = TensorDataset(X_valid_enc, X_valid_dec, Y_valid)
    testset = TensorDataset(X_test_enc, X_test_dec, Y_test)

    train_loader = DataLoader(trainset, batch_size = args['batch_size'], shuffle = True)
    valid_loader = DataLoader(validset, batch_size = args['batch_size'], shuffle = True)
    test_loader = DataLoader(testset, batch_size = 256)

    return train_loader, valid_loader, test_loader


def binary_acc(pred, test):
    pred_tag = torch.round(pred)

    correct_results_sum = (pred_tag == test).sum().float()
    acc = correct_results_sum / test.shape[0]
    return acc


def main(args):
    seed_all(args['seed'])

    model = Model(args)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], betas=(0.9, 0.98), eps=1e-9)
    class_critirion = nn.BCELoss()
    endp_critirion = nn.MSELoss()

    model_folder_name = args['model_name']
    checkpoint_filepath = "checkpoints/{}.pt".format(model_folder_name)
    writer = SummaryWriter('logs/{}'.format(model_folder_name))

    train_loader, valid_loader, test_loader = pre_data(args)

    train(model, train_loader, valid_loader, class_critirion, endp_critirion,
            optimizer, checkpoint_filepath, writer, args)
    
    model = Model(args)
    model.to(device)
    
    checkpoint = torch.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    pred, lab = test(model, test_loader)
    pred_cpu = torch.Tensor.cpu(pred)
    lab_cpu = torch.Tensor.cpu(lab)
    acc = accuracy_score(lab_cpu, np.round(pred_cpu))
    conf_matrix = confusion_matrix(lab_cpu, np.round(pred_cpu), normalize = 'true')
    f1 = f1_score(lab_cpu, np.round(pred_cpu))
    auc = roc_auc_score(lab_cpu, np.round(pred_cpu))

    args['acc'] = acc
    args['f1'] = f1
    args['conf_matrix'] = str(conf_matrix)
    args['auc'] = auc
    #config = json.dumps(args, cls=change_type, indent=4)
    config = json.dumps(args)

    f = open("checkpoints/{}.json".format(model_folder_name),"w")
    f.write(config)
    f.close()

    print(f"Accuracy: {acc} \n AUC: {auc} \n f1: {f1} ")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main(args)


