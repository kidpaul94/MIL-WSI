import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.metrics import roc_auc_score 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import transforms as T

class ClassificationPresetTrain:
    def __init__(self, *, mean=(0.9024, 0.7923, 0.7576), std=(0.1472, 0.2627, 0.3456),):
        trans = [T.ToTensor(), 
                 T.ConvertImageDtype(torch.float),
                 T.Resize((256, 256)),
                 T.Normalize(mean=mean, std=std),
                 T.RandomRotation(degrees=(0, 180)), 
                 T.ColorJitter(brightness=.5, hue=.3), 
                 T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),             
                ]
        self.transforms = T.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

def dataset_mean_std(dataset_path):
    '''This function calculates mean and std for the entire images for training.'''
    dataset = datasets.ImageFolder(dataset_path, transform = T.ToTensor())
    image_loader = DataLoader(dataset, batch_size = 7, shuffle = False, pin_memory = True)

    def get_dimension(loader):
        channel_sum, sum_sq, num_bathces = 0, 0, 0
        
        for data, _ in loader:
            channel_sum += torch.mean(data, dim=[0,2,3])
            sum_sq += torch.mean(data**2, dim=[0,2,3])
            num_bathces += 1
        
        mean = channel_sum / num_bathces
        std = (sum_sq / num_bathces - mean**2)**0.5
        
        return mean, std
        
    mean, std = get_dimension(image_loader)

    return mean, std

def make_tiles(img, tile_size: int = 1024, num_tiles: int = 0):
    '''
    This function divides a large image into samller tiles and removes background
    as much as possible. This function is designed to be used for offline preprocessing of WSI. 
    Run this function to split WSI images into small tiles and save them in a new folder for training.
    Note that some parts of the tissue image can be lost due to the background removal.
    '''
    w, h, ch = img.shape
    pad0, pad1 = (tile_size - w%tile_size) % tile_size, (tile_size - h%tile_size) % tile_size
    padding = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]]
    img = np.pad(img, padding, mode='reflect')
    img = img.reshape(img.shape[0]//tile_size, tile_size, img.shape[1]//tile_size, tile_size, ch)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, ch)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))
    img = img[idxs]
    
    select_idx = []
    
    for idx, tile in enumerate(img):
        gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray,(5,5),0)
        _,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        count = np.count_nonzero(th3) / (1024**2)
        std = np.std(tile)
        
        if count > 0.80 or count < 0.15 or std < 15:
            continue
        else:
            select_idx.append(idx)

    if num_tiles:
        select_idx = select_idx[:num_tiles]

    img = img[select_idx] 
    return img

def tile_track(dataset_path):
    '''
    This function generates .cvs files to store the number of tiles 
    generated from a single training image. The expected format of the tiled images
    is XXX_001.jpg and XXX_010.png.
    '''
    class_folders = [f'{dataset_path}/{f}' for f in listdir(dataset_path)]

    for path in class_folders:
        files = [(f) for f in listdir(path) if isfile(join(path, f))]
        dict = {}

        for file in files:
            if file[:-8] in dict:
                dict[file[:-8]] += 1
            else:
                dict[file[:-8]] = 1

        with open(f'{path}.csv', 'w') as f:
            f.write('%s, %s\n' % ('image_id', 'patches'))
            for key in dict.keys():
                f.write("%s, %s\n" % (key, dict[key]))

def accuracy(output, target):
    '''Accuracy calculation using L1 distance'''
    pred = nn.Softmax(dim = 1)(output).detach().numpy()
    target = target.detach().numpy()
    
    return np.linalg.norm(pred-target, ord=1), pred[0]

def eval_auc(y_true, y_prob):
    '''ROC_AUC'''
    y_true = np.asarray(y_true).astype(np.float32)
    y_prob = np.asarray(y_prob).astype(np.float32)

    return roc_auc_score(y_true, y_prob)