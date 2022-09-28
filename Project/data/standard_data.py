import torch
import os
import numpy as np
import pickle5 as pk
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms as A

from tqdm import tqdm
from pie_data import PIE


class StandardData(data.Dataset):
    def __init__(self, path, pie_path, data_set, args, balance=False,
                 bn='all', t23=False):
        np.random.seed(args.seed)
        self.max_w_var = 9
        self.max_h_var = 6
        self.max_d_var = 2
        self.input_size = int(32*1)
        if data_set == 'train':
            num_samples = [9974, 5956, 7867]
        elif data_set == 'test':
            num_samples = [9921, 5346, 3700]
        elif data_set == 'val':
            num_samples = [3404, 1369, 1813]

        balance_data = [max(num_samples) / s for s in num_samples]
        if data_set == 'test':
            if bn != 'all':
                balance_data[2] = 0
            elif t23:
                balance_data = [1, (num_samples[0] + num_samples[2]) / num_samples[1], 1]

        self.data_path = os.getcwd() / Path(path) / 'data'
        self.img_path = os.getcwd() / Path(path) / 'imgs'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        self.pie_path = pie_path

        imdb = PIE(data_path=self.pie_path)
        params = {'data_split_type': 'default', }
        self.vid_ids, _ = imdb._get_data_ids(data_set, params)

        filt_list = lambda x: x.split('_')[0] in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))

        self.ped_data = {}
        ped_ids = ped_ids[:1000]

        for ped_id in tqdm(ped_ids, desc=f'loading {data_set} data in memory'):
            ped_path = self.data_path.join(ped_id).as_posix()
            loaded_data = self.load_data(ped_path)
            img_file = str(self.imgs_path / loaded_data['crop_img'].stem) + '.pkl'
            loaded_data['crop_img'] = self.load_data(img_file)

            if loaded_data['irr'] == 1 and bn != 'all':
                continue

            if balance:
                if loaded_data['irr'] == 1:
                    self.repeat_data(balance_data[2], loaded_data, ped_id)
                elif loaded_data['crossing'] == 0:
                    self.repeat_data(balance_data[0], loaded_data, ped_id)
                elif loaded_data['crossing'] == 1:
                    self.repeat_data(balance_data[1], loaded_data, ped_id)

            else:
                self.ped_data[ped_id.split('.')[0]] = loaded_data

        self.ped_ids = list(self.ped_data.keys())
        self.data_len = len(self.ped_ids)

    def repeat_data(self, n_rep, data, ped_id):
        ped_id = ped_id.split('.')[0]

        if self.data_set == 'train' or self.data_set == 'val' or self.t23:
            prov = n_rep % 1
            n_rep = int(n_rep) if prov == 0 else int(n_rep) + np.random.choice(2, 1, p=[1-prov, prov])[0]
        else:
            n_rep = int(n_rep)

        for i in range(int(n_rep)):
            self.ped_data[ped_id + f'-r{i}'] = data

    def load_data(self, data_path):
        with open(data_path, 'rb') as fid:
            database = pk.load(fid, encoding='bytes')
        return database

    def transforms(self, img):
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
        return transform(img)


    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        ped_id = self.ped_ids[idx]
        ped_data = self.ped_data[ped_id]
        w, h = ped_data['w'], ped_data['h']

        ped_data['kps'][-30:] = ped_data['kps_forecast']
        kp = ped_data['kps']
        if self.data_set == 'train':
            kp[..., 0] = np.clip(kp[..., 0] + np.random.randint(self.max_w_var, size=kp[..., 0].shape), 0, w)
            kp[..., 1] = np.clip(kp[..., 1] + np.random.randint(self.max_h_var, size=kp[..., 1].shape), 0, w)
            kp[..., 2] = np.clip(kp[..., 2] + np.random.randint(self.max_d_var, size=kp[..., 2].shape), 0, w)

        kp[..., 0] /= w
        kp[..., 1] /= h
        kp[..., 2] /= 80

        kp = torch.from_numpy(kp.transpose(2, 0, 1).float().contiguous())

        seg_map = torch.from_numpy(ped_data['crop_img'][:1]).float()
        seg_map = (seg_map - 78.26) / 45.12
        img = ped_data['crop_img'][1:]
        img = self.transforms(img.transpose(1, 2, 0)).contiguous()
        img = torch.cat([seg_map, img], 0)

        vel_obd = np.asarray(ped_data['obd_speed']).reshape(1, -1) / 120.0
        vel_gps = np.asarray(ped_data['gps_speed']).reshape(1, -1) / 120.0
        vel = torch.from_numpy(np.concatenate([vel_gps, vel_obd], 0)).float().contiguous()

        if ped_data['irr']:
            bh = torch.from_numpy(np.ones(1).reshape([1])) * 2
        else:
            bh = torch.from_numpy(ped_data['crossing'].reshape([1])).float()

        return kp, bh, img, vel

