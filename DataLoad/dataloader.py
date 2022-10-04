import torch
import torch.utils.data as data
import os
import pickle5 as pk
from pathlib import Path
import numpy as np
from torchvision import transforms as A
from tqdm import tqdm, trange

from pie_data import PIE


class DataSet(data.Dataset):
    def __init__(self, path, pie_path, data_set):
        super().__init__()
        np.random.seed(42)

        self.data_set = data_set
        self.maxw_var = 9
        self.maxh_var = 6
        self.maxd_var = 2

        self.data_path = os.getcwd() / Path(path) / 'data'
        self.img_path = os.getcwd() / Path(path) / 'imgs'
        self.data_list = [data_name for data_name in os.listdir(self.data_path)]
        self.pie_path = pie_path

        imdb = PIE(data_path=self.data_path)
        params = {'data_split_type': 'default'}
        self.vid_ids, _ = imdb._get_data_ids(data_set, params)

        filt_list = lambda x: x.split('_')[0] in self.vid_ids
        ped_ids = list(filter(filt_list, self.data_list))

        self.ped_data = {}
        # ped_ids = ped_ids[: 1000]

        for ped_id in tqdm(ped_ids, desc=f'loading {data_set} data in memory'):
            ped_path = self.data_path.joinpath(ped_id).as_posix()
            loaded_data = self.load_data(ped_path)
            # img_file = str(self.img_path / loaded_data['crop_img'].stem) + '.pkl'
            # loaded_data['crop_img'] = self.load_data(img_file)

            self.ped_data[ped_id.split('.')[0]] = loaded_data

        self.ped_ids = list(self.ped_data.keys())
        self.data_len = len(self.ped_ids)

    def load_data(self, data_path):
        with open(data_path, 'rb') as fid:
            database = pk.load(fid, encoding='bytes')
        return database

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        ped_id = self.ped_ids[index]
        ped_data = self.ped_data[ped_id]
        w, h = ped_data['w'], ped_data['h']

        kp = ped_data['kps'][:-30]
        if self.data_set == 'train':
            kp[..., 0] = np.clip(kp[..., 0] + np.random.randint(self.maxw_var, size=kp[..., 0].shape), 0, w)
            kp[..., 1] = np.clip(kp[..., 1] + np.random.randint(self.maxh_var, size=kp[..., 1].shape), 0, w)
            kp[..., 2] = np.clip(kp[..., 2] + np.random.randint(self.maxd_var, size=kp[..., 2].shape), 0, w)

        kp[..., 0] /= w
        kp[..., 1] /= h
        kp[..., 2] /= 80
        kp = torch.from_numpy(kp.transpose(2, 0, 1)).float().contiguous()

        return kp


def main():
    data_path = './data/PIE'
    pie_path = './PIE'

    transform = A.Compose(
        [
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    tr_data = DataSet(path=data_path, pie_path=pie_path, data_set='train')
    iter_ = trange(len(tr_data))
    cx = np.zeros([len(tr_data), 3])
    fs = np.zeros([len(tr_data), 192, 64])

    for i in iter_:
        x, y, f, v = tr_data.__getitem__(i)

        # y = np.clip(y - 1, 0, 1)
        # y[y==2] = 0
        fs[i] = f[0]
        cx[i, y.long().item()] = 1

    print(f'No Crosing: {cx.sum(0)[0]} Crosing: {cx.sum(0)[1]}, Irrelevant: {cx.sum(0)[2]} ')
    print('finish')


if __name__ == "__main__":
    main()
