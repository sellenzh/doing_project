import torch
import torch.utils.data as data
import os
import pickle5 as pk
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from torchvision import transforms as A
import torch.nn.functional as F

from tqdm import tqdm, trange
from regen_jaad import JAAD


class DataSet(data.Dataset):
    def __init__(self, path, jaad_path, balance=True, bh='all', transforms=None, pcpa=None, forecast=True):

        np.random.seed(42)
        self.forecast = forecast
        self.bh = bh
        self.pcpa = os.getcwd() / Path(pcpa)
        self.transforms = transforms
        self.balance = balance
        self.data_set = 'test'
        self.last2 = False
        self.maxw_var = 9
        self.maxh_var = 6
        self.maxd_var = 2
        self.input_size = int(32 * 1)
        
        nsamples = [1871, 3204, 13037]
        balance_data = [max(nsamples) / s for s in nsamples]

        self.data_path = os.getcwd() / Path(path) / 'data'
        self.imgs_path = os.getcwd() / Path(path) / 'imgs'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        self.jaad_path = jaad_path
        
        imdb = JAAD(data_path=self.jaad_path)
        self.vid_ids = imdb._get_video_ids_split(self.data_set)
        
        filt_list =  lambda x: "_".join(x.split('_')[:2]) in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))

        if bh != 'all':
            filt_list =  lambda x: 'b' in x 
            ped_ids = list(filter(filt_list, ped_ids))

        pcpa_, dense_, fussi_ = self.load_3part()
        
        self.models_data = {}

        for k_id in pcpa_['ped_id'].keys():
            vid_n = int(k_id.split('_')[1])
            vid_n = f'video_{vid_n:04}' 
            ped_id = k_id.split('fr')[0]
            frm = k_id.split('fr')[1]

            pcpa_key =  vid_n + '_pid_' + ped_id + '_fr' + frm
            try:
                dense_res = dense_['result'][k_id]
                dense_lab = dense_['labels'][k_id]
                fussi_res = fussi_['result'][k_id]
                fussi_lab = fussi_['labels'][k_id]
                pcpa_res = pcpa_['result'][k_id]
                pcpa_lab = pcpa_['labels'][k_id]
                assert dense_lab == fussi_lab == pcpa_lab
            except KeyError:
                continue

            pcpa_dict = {'result': pcpa_res, 'label': pcpa_lab}
            dense_dict = {'result': dense_res, 'label': dense_lab}
            fussi_dict = {'result': fussi_res, 'label': fussi_lab}
            if pcpa_key + '.pkl' in ped_ids:
                self.models_data[pcpa_key] = [pcpa_dict, dense_dict, fussi_dict]
        list_k = list(self.models_data.keys())

        filt_list =  lambda x: x.split('.')[0] in list_k
        ped_ids = list(filter(filt_list, ped_ids))

        

        self.ped_data = {}
        for ped_id in tqdm(ped_ids, desc=f'loading {self.data_set} data in memory'):
            
            ped_path = self.data_path.joinpath(ped_id).as_posix()
            loaded_data = self.load_data(ped_path)
            img_file = str(self.imgs_path / loaded_data['crop_img'].stem) + '.pkl'
            loaded_data['crop_img'] = self.load_data(img_file)

            if balance:
                if 'b' not in ped_id:               # irrelevant
                    self.repet_data(balance_data[2], loaded_data, ped_id)
                elif loaded_data['crossing'] == 0:  # no crossing
                    self.repet_data(balance_data[0], loaded_data, ped_id)
                elif loaded_data['crossing'] == 1:  # crossing
                    self.repet_data(balance_data[1], loaded_data, ped_id)  
            else:
                
                self.ped_data[ped_id.split('.')[0]] = loaded_data
                
             
        self.ped_ids = list(self.ped_data.keys())
        self.data_len = len(self.ped_ids)
    
    def load_3part(self, ):
        pcpa, fussi, g_pcpca = {}, {}, {}
        mod = 'bh' if self.bh == 'bh' else 'all'
        
        if self.last2:
            pcpa['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_preds_jaad_{mod}_last2.pkl')
            pcpa['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_labels_jaad_{mod}_last2.pkl')
            pcpa['ped_id'] = self.load_data(self.pcpa.__str__() + f'/test_results/jaad/pcpa_ped_ids_jaad_{mod}_last2.pkl')
            
            fussi['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_preds_jaad_last2.pkl')
            fussi['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_labels_jaad_last2.pkl')
            fussi['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_ped_ids_jaad_last2.pkl')
            
            g_pcpca['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpa_preds_jaad_{mod}_last2.pkl')
            g_pcpca['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpa_labels_jaad_{mod}_last2.pkl')
            g_pcpca['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpa_ped_ids_jaad_{mod}_last2.pkl')
        
        else:
            pcpa['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_preds_jaad_{mod}.pkl')
            pcpa['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_labels_jaad_{mod}.pkl')
            pcpa['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/pcpa_ped_ids_jaad_{mod}.pkl')

            fussi['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_preds_jaad.pkl')
            fussi['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_labels_jaad.pkl')
            fussi['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/fussi_ped_ids_jaad.pkl')

            g_pcpca['result'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpca_preds_jaad_{mod}.pkl')
            g_pcpca['labels'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpca_labels_jaad_{mod}.pkl')
            g_pcpca['ped_id'] = self.load_data(self.pcpa.__str__() + f'/jaad/g_pcpca_ped_ids_jaad_{mod}.pkl')
        
        return pcpa, fussi, g_pcpca

    def repet_data(self, n_rep, data, ped_id):
        ped_id = ped_id.split('.')[0]

        if self.data_set == 'train' or self.data_set == 'val':
            prov = n_rep % 1  
            n_rep = int(n_rep) if prov == 0 else int(n_rep) + np.random.choice(2, 1, p=[1 - prov, prov])[0]
            # n_rep = int(n_rep * 2)
        else:
            n_rep = int(n_rep)

        for i in range(n_rep):
            self.ped_data[ped_id + f'-r{i}'] = data

    
    def load_data(self, data_path):

        with open(data_path, 'rb') as fid:
            database = pk.load(fid, encoding='bytes')
        return database
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, item):
        
        
        ped_id = self.ped_ids[item]
        pcpa_dict, dense_dict, g_pcpca_dict = self.models_data[ped_id.split('.')[0].split('-')[0]]
        models_data = {'pcpa': pcpa_dict, 'fussi': dense_dict, 'g_pcpca': g_pcpca_dict}
        ped_data = self.ped_data[ped_id]

        weather_ = ped_data['weather'] 
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
        return keypoints, bh, img, bbox, vel, cross_point, weather_, models_data
