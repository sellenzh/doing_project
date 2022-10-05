import time
from utils.pie_data import PIE
from utils.preprocessing import *
from utils.data_loader import DataSet
from utils.train_utils import train, test
from model.model import Model


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import json

seed = 42
seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIE_PATH = 'PIE_dataset'

input_opts = {'num_layers' : 4,
              'd_model': 128,
              'd_input':4,
              'num_heads' : 8,
              'dff': 256,
              'pos_encoding': 16,
              'rate': 0.3,
              'batch_size': 32,
              'warmup_steps': 1000,
              'model_name' : time.strftime("%d%b%Y-%Hh%Mm%Ss"),
              'pooling' : False,
              'optimizer': 'Adam',
              'tte' : [30, 60],
              'seed':seed
        }

train_dataset = DataSet(path='./data/PIE', pie_path='./PIE', data_set='train')
valid_dataset = DataSet(path='./data/PIE', pie_path='./PIE', data_set='val')
test_dataset = DataSet(path='./data/PIE', pie_path='./PIE', data_set='test')

pose_tr, pose_val = torch.Tensor(train_dataset[0]), torch.Tensor(valid_dataset[0])
y_tr, y_val = torch.Tensor(train_dataset[-1]), torch.Tensor(valid_dataset[-1])
pose_te, y_te = torch.Tensor(test_dataset[0]), torch.Tensor(test_dataset[-1])

train_set = TensorDataset(pose_tr, y_tr)
valid_set = TensorDataset(pose_val, y_val)
test_set = TensorDataset(pose_te, y_te)

train_loader = DataLoader(train_set, batch_size=input_opts['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=input_opts['batch_size'], shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)


#Training Loop
print("Start Training Loop......\n")
epochs = 200

model = Model(num_layers=input_opts['num_layers'], d_model=input_opts['d_model'],
              d_input=input_opts['d_input'], num_heads=input_opts['num_heads'],
              dff=input_opts['dff'], maximum_position_encoding=input_opts['pos_encoding'],
              rate=input_opts['rate'])

model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.BCELoss()

model_folder_name = 'Encoder_Only__' + input_opts['model_name']
checkpoint_filepath = "checkpoints/{}.pt".format(model_folder_name)
writer = SummaryWriter('logs/{}'.format(model_folder_name))

train(model, train_loader, valid_loader, criterion, optimizer, checkpoint_filepath, writer, epochs)

#Test
model = Model(num_layers=input_opts['num_layers'], d_model=input_opts['d_model'],
              d_input=input_opts['d_input'], num_heads=input_opts['num_heads'],
              dff=input_opts['dff'], maximum_position_encoding=input_opts['pos_encoding'])

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

input_opts['acc'] = acc
input_opts['f1'] = f1
input_opts['conf_matrix'] = str(conf_matrix)
input_opts['auc'] = auc
config = json.dumps(input_opts)


f = open("checkpoints/{}.json".format(model_folder_name),"w")
f.write(config)
f.close()

print(f"Accuracy: {acc} \n AUC: {auc} \n f1: {f1}")

