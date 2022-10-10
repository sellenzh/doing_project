from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from model.model import Model
from tools.pie_data import PIE
from tools.preprocessing import *
from tools.train_tools import train, test


seed = 42
seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

PIE_PATH = './PIE_dataset'
data_opts = {
    'fstride': 1,
    'sample_type': 'all',
    'height_rng': [0, float('inf')],
    'squarify_ratio': 0,
    'data_split_type': 'default',  # kfold, random, default
    'seq_type': 'crossing',  # crossing , intention
    'min_track_size': 15,  # discard tracks that are shorter
    'kfold_params': {'num_folds': 1, 'fold': 1},
    'random_params': {'ratios': [0.7, 0.15, 0.15],
                    'val_data': True,
                    'regen_data': False},
    'tte': [30, 60]
}
input_opts = {
    'num_layers': 4,
    'd_model': 128,
    'bbox_input': 4,
    'speed_input': 2,
    'num_heads': 8,
    'dff': 256,
    'pos_encoding': 16,
    'batch_size': 32,
    'warmup_steps': 1000,
    'model_name': 'bbox_vel',
    'pooling': False,
    'optimizer': 'Adam',
    'tte': data_opts['tte'],
    'seed': seed
}

imdb = PIE(data_path=PIE_PATH)
seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
balanced_seq_train = balance_dataset(seq_train)
tte_seq_train, _ = tte_dataset(balanced_seq_train, data_opts['tte'], 0.6, 16)

seq_valid = imdb.generate_data_trajectory_sequence('val', **data_opts)
balanced_seq_valid = balance_dataset(seq_valid)
tte_seq_valid, _ = tte_dataset(balanced_seq_valid, data_opts['tte'], 0, 16)

seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
tte_seq_test, _ = tte_dataset(seq_test, data_opts['tte'], 0, 16)

bbox_train = tte_seq_train['bbox']
bbox_valid = tte_seq_valid['bbox']
bbox_test = tte_seq_test['bbox']

action_train = tte_seq_train['activities']
action_valid = tte_seq_valid['activities']
action_test = tte_seq_test['activities']

vel_train = torch.cat((tte_seq_train['obd_speed'], tte_seq_train['gps_speed']), dim=-1)
vel_valid = torch.cat((tte_seq_valid['obd_speed'], tte_seq_valid['gps_speed']), dim=-1)
vel_test = torch.cat((tte_seq_test['obd_speed'], tte_seq_test['gps_speed']), dim=-1)


normalized_bbox_train = normalize_bbox(bbox_train)
normalized_bbox_valid = normalize_bbox(bbox_valid)
normalized_bbox_test = normalize_bbox(bbox_test)

label_action_train = prepare_label(action_train)
label_action_valid = prepare_label(action_valid)
label_action_test = prepare_label(action_test)

X_train, X_valid = torch.Tensor(normalized_bbox_train), torch.Tensor(normalized_bbox_valid)
Y_train, Y_valid = torch.Tensor(label_action_train), torch.Tensor(label_action_valid)
X_test = torch.Tensor(normalized_bbox_test)
Y_test = torch.Tensor(label_action_test)

train_set = TensorDataset(X_train, vel_train, Y_train)
valid_set = TensorDataset(X_valid, vel_valid, Y_valid)
test_set = TensorDataset(X_test, vel_test, Y_test)

train_loader = DataLoader(train_set, batch_size=input_opts['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=input_opts['batch_size'], shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

# Training Loop
print("Start Training Loop \n")
epochs = 200

model = Model(num_layers=input_opts['num_layers'],
                d_model=input_opts['d_model'],
                bbox_input=input_opts['bbox_input'],
                speed_input=input_opts['speed_input'],
                num_heads=input_opts['num_heads'],
                dff=input_opts['dff'],
                maximum_position_encoding=input_opts['pos_encoding'])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.BCELoss()

model_folder_name = 'Encoder_Only_' + input_opts['model_name']
checkpoint_filepath = "checkpoints/{}.pt".format(model_folder_name)
writer = SummaryWriter('logs/{}'.format(model_folder_name))

train(model, train_loader, valid_loader, criterion, optimizer, checkpoint_filepath, writer, epochs)

# Test
model = Model(num_layers=input_opts['num_layers'],
                d_model=input_opts['d_model'],
                bbox_input=input_opts['bbox_input'],
                speed_input=input_opts['speed_input'],
                num_heads=input_opts['num_heads'],
                dff=input_opts['dff'],
                maximum_position_encoding=input_opts['pos_encoding'])
model.to(device)
checkpoint = torch.load(checkpoint_filepath)
model.load_state_dict(checkpoint['model_state_dict'])

pred, lab = test(model, test_loader)

pred_cpu = torch.Tensor.cpu(pred)
lab_cpu = torch.Tensor.cpu(lab)
acc = accuracy_score(lab_cpu, np.round(pred_cpu))
conf_matrix = confusion_matrix(lab_cpu, np.round(pred_cpu), normalize='true')
f1 = f1_score(lab_cpu, np.round(pred_cpu))
auc = roc_auc_score(lab_cpu, np.round(pred_cpu))

input_opts['acc'] = acc
input_opts['f1'] = f1
input_opts['conf_matrix'] = str(conf_matrix)
input_opts['auc'] = auc
config = json.dumps(input_opts)

f = open("checkpoints/{}.json".format(model_folder_name), "w")
f.write(config)
f.close()

print(f"Accuracy: {acc} \n AUC: {auc} \n f1: {f1}")



