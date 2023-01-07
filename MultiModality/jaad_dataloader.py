import torch
import torch.utils.data as data
import os
import pickle5 as pk
from pathlib import Path
import numpy as np
from torchvision import transforms as A

from tqdm import tqdm, trange
from regen_jaad import JAAD

class DataSet(data.Dataset):
    def __init__(self, path, jaad_path, data_set, balance=True, bh='all', transforms=None, pcpa=None, forcast=False):
        super().__init__()
        np.random.seed(42)
        self.balance = balance
        self.data_set = data_set
        self.transforms = transforms
        self.seg = True
        self.maxw_var = 9
        self.maxh_var = 6
        self.maxd_var = 2
        if not pcpa is None:
            self.pcpa = Path(pcpa)
        self.forecast = forcast

        if data_set == 'train':
            num_samples = [1025, 4778, 17582]
        elif data_set == 'test':
            num_samples = [1871, 3204, 13037]
        elif data_set == 'val':
            num_samples = [176, 454, 2772]
        balance_data = [max(num_samples) / s for s in num_samples]
        self.data_path = os.getcwd() / Path(path) / 'data'
        self.imgs_path = os.getcwd() / Path(path) / 'imgs'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        self.jaad_path = jaad_path

        imdb = JAAD(data_path=self.jaad_path)
        self.vid_ids = imdb._get_video_ids_split(data_set)
        filt_list = lambda x: '_'.join(x.split('_')[:2]) in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))
        if bh != 'all':
            filt_list = lambda x: 'b' in x
            ped_ids = list(filter(filt_list, ped_ids))
        if data_set == 'test' and not pcpa is None:
            pcpa_res = self.load_data(self.pcpa / 'test_res_jaad.pkl')
            pcpa_lab = self.load_data(self.pcpa / 'test_labels_jaad.pkl')
            pcpa_ids = self.load_data(self.pcpa / 'ped_ids_jaad.pkl')
            self.pcpa_data = {}

            for id_num in range(len(pcpa_ids)):
                vid_n, frm, ped_id = pcpa_ids[id_num].split('-')
                pcpa_key =  vid_n + '_pid_' + ped_id + '_fr_' + frm
                
                if pcpa_key + '.pkl' in ped_ids:
                    self.pcpa_data[pcpa_key] = [pcpa_res[id_num], pcpa_lab[id_num]]
            list_k = list(self.pcpa_data.keys())

            filt_list =  lambda x: x.split('.')[0] in list_k
            ped_ids = list(filter(filt_list, ped_ids))
        self.ped_data = {}
        for ped_id in tqdm(ped_ids, desc=f'loading {data_set} data in memory'):
            ped_path = self.data_path.joinpath(ped_id).as_posix()
            loaded_data = self.load_data(ped_path)

            img_file = self.imgs_path.joinpath(loaded_data['crop_img'].stem + '.pkl').as_posix()
            loaded_data['crop_img'] = self.load_data(img_file)

            if balance:
                if 'b' not in ped_id:
                    self.repet_data(balance_data[2], loaded_data, ped_id)
                elif loaded_data['crossing'] == 0:
                    self.repet_data(balance_data[0], loaded_data, ped_id)
                elif loaded_data['crossing'] == 1:
                    self.repet_data(balance_data[1], loaded_data, ped_id)
            else:
                self.ped_data[ped_id.split('.')[0]] = loaded_data
        self.ped_ids = list(self.ped_data.keys())
        self.data_len = len(self.ped_ids)


    def load_data(self, data_path):
        with open(data_path, 'rb') as fid:
            database = pk.load(fid, encoding='bytes')
        return database
    
    def repet_data(self, n_rep, data, ped_id):
        ped_id = ped_id.split('.')[0]
        if self.data_set == 'train' or self.data_set == 'val':
            prov = n_rep % 1
            n_rep = int(n_rep) if prov == 0 else int(n_rep) + np.random.choice(2, 1, p=[1-prov, prov])[0]
        else:
            n_rep = int(n_rep)
        for i in range(n_rep):
            self.ped_data[ped_id + f'-r{i}'] = data

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        ped_id = self.ped_ids[index]
        ped_data = self.ped_data[ped_id]
        w, h = ped_data['w'], ped_data['h']

        if self.forecast:
            ped_data['kps'][-30:] = ped_data['kps_forcast']
            keypoints = ped_data['kps']
        else:
            keypoints = ped_data['kps'][:-30]
        if self.data_set == 'train':
            keypoints[..., 0] = np.clip(keypoints[..., 0] + np.random.randint(self.maxw_var, size=keypoints[..., 0].shape), 0, w)
            keypoints[..., 1] = np.clip(keypoints[..., 1] + np.random.randint(self.maxh_var, size=keypoints[..., 1].shape), 0, h)
            keypoints[..., 2] = np.clip(keypoints[..., 2] + np.random.randint(self.maxd_var, size=keypoints[..., 2].shape), 0, 80)
        keypoints[..., 0] /= w
        keypoints[..., 1] /= h
        keypoints[..., 2] /= 80
        keypoints = torch.from_numpy(keypoints.transpose(2, 0, 1)).float().contiguous()

        seg_map = torch.from_numpy(ped_data['crop_img'][:1]).float() 
        seg_map = (seg_map - 78.26) / 45.12
        img = ped_data['crop_img'][1:].transpose(1, 2, 0).copy()
        img = self.transforms(img).contiguous()
        if self.seg:
            img = torch.cat([seg_map, img], 0)
        
        vel = torch.from_numpy(np.tile(ped_data['vehicle_act'], [1, 2]).transpose(1, 0)).float().contiguous()
        vel = vel[:, :-30] # assert only 32 samples 

        bbox = np.array(ped_data['bbox'])
        bbox[:2, :] /= 1920
        bbox[2:, :] /= 1080
        bbox = torch.from_numpy(bbox).transpose(-1, -2).float().contiguous()

        cross_point = np.array(ped_data['cross_point'])
        cross_point[:2] /= 1920
        cross_point[2:] /= 1080
        cross_point = torch.from_numpy(cross_point).float().contiguous()

        # 0 for no crossing,  1 for crossing, 2 for irrelevant
        idx = -2 if self.balance else -1 
        if 'b' not in ped_id.split('-')[idx]: # if is irrelevant
            bh = torch.from_numpy(np.ones(1).reshape([1])) * 2 
            #bh = torch.from_numpy(np.zeros(1).reshape([1]))
        else:                               # if is crosing or not
            bh = torch.from_numpy(ped_data['crossing'].reshape([1])).float()
        return keypoints, bh, img, bbox, vel, cross_point
 
