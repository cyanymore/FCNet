import os
import torch
import glob
import os.path as osp
# from model import Generator, GNN
from FCNet import Generator, SGA
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
G = Generator()
GNN = SGA(channel=1472)


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class DataSet(data.Dataset):
    def __init__(self, img_transform_gt, img_transform_sketch, dataset_name):
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch

        self.img_dir_s = r'F:\RefDataset\KAIST\test\refA'
        self.img_dir_r = r'F:\RefDataset\KAIST\test\refB'

        self.ref_name = glob.glob(os.path.join(self.img_dir_r, '*.*'))

        self.skt_name = glob.glob(os.path.join(self.img_dir_s, '*.*'))
        # print(self.ref_name)
        # print(self.skt_name)

        self.img_size = (256, 256, 3)

    def __getitem__(self, index):
        reference = Image.open(self.ref_name[index]).convert('RGB')
        sketch = Image.open(self.skt_name[index]).convert('L')

        return self.img_transform_gt(reference), self.img_transform_sketch(sketch)

    def __len__(self):
        """Return the number of images."""
        return len(self.ref_name)  # 或者使用 len(self.skt_name)


def get_loader(dataset_name):
    img_transform_gt = list()
    img_transform_sketch = list()
    img_size = 256

    img_transform_gt.append(T.Resize((img_size, img_size)))
    img_transform_gt.append(T.ToTensor())
    img_transform_gt.append(T.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform_gt = T.Compose(img_transform_gt)

    img_transform_sketch.append(T.Resize((img_size, img_size)))
    img_transform_sketch.append(T.ToTensor())
    img_transform_sketch.append(T.Normalize(mean=0.5, std=0.5))
    img_transform_sketch = T.Compose(img_transform_sketch)

    dataset = DataSet(img_transform_gt, img_transform_sketch, dataset_name)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  num_workers=1)
    # print(len(data_loader))
    return data_loader


def image_save(gen, fid, sample_dir):
    sample_path = sample_dir + '/' + fid + '.png'
    save_image(denorm(gen.data.cpu()), sample_path, nrow=1, padding=0)


def load_model(dataset_name, epoch):
    G_path = './' + dataset_name + '/models/{}-G.pth'.format(epoch)
    # G_checkpoint = torch.load(G_path)
    G_checkpoint = torch.load(G_path, map_location=device)
    G.load_state_dict(G_checkpoint['model'])
    # G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    G.to(device)
    G.eval()


if __name__ == '__main__':
    dataset_name = 'nighttime_k'
    test_loader = get_loader(dataset_name)

    load_model(dataset_name, 99)
    test_iter = iter(test_loader)
    for i in range(len(test_loader)):
        ref, skt = next(test_iter)
        ref = ref.to(device)
        skt = skt.to(device)
        result = G(ref, skt)
        image_save(result, str(i).zfill(5), './' + dataset_name + '/exp')
        image_save(ref, str(i).zfill(5), './' + dataset_name + '/ref')
