# modified from https://github.com/lorenmt/mtan/blob/master/im2im_pred/create_dataset.py

from torch.utils.data.dataset import Dataset, DataLoader
import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        return img_, label_, depth_ / sc, normal_


class NYUv2(Dataset):
    """
    We could further improve the performance with the data augmentation of NYUv2 defined in:
        [1] PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing
        [2] Pattern affinitive propagation across depth, surface normal and semantic segmentation
        [3] Mti-net: Multiscale task interaction networks for multi-task learning

        1. Random scale in a selected raio 1.0, 1.2, and 1.5.
        2. Random horizontal flip.

    Please note that: all baselines and MTAN did NOT apply data augmentation in the original paper.
    """
    def __init__(self, root, mode='train', augmentation=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        
        random.seed(688)
        np.random.seed(688)
        data_len = len(fnmatch.filter(os.listdir(self.root + '/train/image'), '*.npy'))
        train_index = list(np.random.choice(np.arange(data_len), size=int(data_len*0.8), replace=False))
        val_index = list(set(range(data_len)).difference(set(train_index)))
        # read the data file
        if self.mode == 'train':
            self.index_list = train_index
            self.data_path = self.root + '/train'
        elif self.mode == 'val':
            self.index_list = val_index
            self.data_path = self.root + '/train'
        elif self.mode == 'trainval':
            self.index_list = train_index + val_index
            self.data_path = self.root + '/train'
        elif self.mode == 'test':
            data_len = len(fnmatch.filter(os.listdir(self.root + '/val/image'), '*.npy'))
            self.index_list = list(range(data_len))
            self.data_path = self.root + '/val'

        # calculate data length
#         self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, i):
        index = self.index_list[i]
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))

        # apply data augmentation if required
        if self.augmentation:
            image, semantic, depth, normal = RandomScaleCrop()(image, semantic, depth, normal)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                normal[0, :, :] = - normal[0, :, :]

        return image.float(), semantic.float(), depth.float(), normal.float()

    def __len__(self):
        return len(self.index_list)

############################


class office_Dataset(Dataset):
    def __init__(self, dataset, task, mode, dataroot):
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                        ])
        if mode != 'trval':
            f = open('./office_data/{}/{}_{}.txt'.format(dataset, task, mode), 'r')
            self.img_list = f.readlines()
            f.close()
        else:
            f1 = open('./office_data/{}/{}_train.txt'.format(dataset, task), 'r')
            f2 = open('./office_data/{}/{}_val.txt'.format(dataset, task), 'r')
            self.img_list = f1.readlines() + f2.readlines()
            f1.close()
            f2.close()
        self.root_path = dataroot
        
    def __getitem__(self, i):
        img_path = self.img_list[i][:-1].split(' ')[0]
        y = int(self.img_list[i][:-1].split(' ')[1])
        img = Image.open(self.root_path + img_path).convert('RGB')
        return self.transform(img), y
        
    def __len__(self):
        return len(self.img_list)
    
def office_dataloader(dataset, batchsize):
    if dataset == 'office-31':
        tasks = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-home':
        tasks = ['Art', 'Clipart', 'Product', 'Real_World']
    data_loader = {}
    iter_data_loader = {}
    for k, d in enumerate(tasks):
        data_loader[k] = {}
        iter_data_loader[k] = {}
        for mode in ['train', 'val', 'test', 'trval']:
            shuffle = False if mode == 'test' else True
            drop_last = False if mode == 'test' else True
            txt_dataset = office_Dataset(dataset, d, mode, dataroot)
            print(d, mode, len(txt_dataset))
            data_loader[k][mode] = DataLoader(txt_dataset, 
                                              num_workers=0, 
                                              pin_memory=True, 
                                              batch_size=batchsize, 
                                              shuffle=shuffle,
                                              drop_last=drop_last)
            iter_data_loader[k][mode] = iter(data_loader[k][mode])
    return data_loader, iter_data_loader
