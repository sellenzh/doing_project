from torchvision import transforms as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pie_dataloader23 import DataSet


class DInterface(pl.LightningDataModule):
    def __init__(self, args, num_workers=8, dataset='',  **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = args.batch_size

        transform = A.Compose(
            [A.ToPILImage(),
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

        self.trainset = DataSet(path=args.data_path, pie_path=args.data_path, data_set='train', frame=True, vel=True,
                                balance=False, transforms=transform, seg_map=args.seg, h3d=args.H3D, forecast=args.forecast)
        self.testset = DataSet(path=args.data_path, pie_path=args.pie_path, data_set='test', frame=True, vel=True, balance=False,
                      transforms=transform, seg_map=args.seg, h3d=args.H3D, t23=False, forecast=args.forecast)
        self.valset = DataSet(path=args.data_path, pie_path=args.pie_path, data_set='val', frame=True, vel=True, balance=False,
                       transforms=transform, seg_map=args.seg, h3d=args.H3D, forecast=args.forecast)


    def train_dataloader(self):
        return  DataLoader(self.trainset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=True,
                           pin_memory=True)

    def val_dataloader(self):
        return  DataLoader(self.valset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False,
                           pin_memory=True)

    def test_dataloader(self):
        return  DataLoader(self.testset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=False,
                           pin_memory=True)

    def forward(self):
        return self.train_dataloader(), self.test_dataloader(), self.val_dataloader()

