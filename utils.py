import numpy as np
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

def accuracy(output, target):    
    pred = nn.Softmax(dim = 1)(output).detach().numpy()
    target = target.detach().numpy()
    
    return np.linalg.norm(pred-target, ord=1), pred[0] # L1 distance

def eval_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(np.float32)
    y_prob = np.asarray(y_prob).astype(np.float32)

    return roc_auc_score(y_true, y_prob) # ROC_AUC 