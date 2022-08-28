import gc
import random
import cv2 as cv
import tqdm as tqdm

import torch
import torch.nn as nn

from utils import ClassificationPresetTrain, accuracy, eval_auc

def validation(model, criterion, bag_size, test_path, transform):
    model.eval()
    softmax = nn.Softmax(dim = 1)
    print("Prediction...")

def engine(model, device, criterion, optimizer, lr_scheduler, iterate, batch_size, bag_size, num_data, scaler):

    transform = ClassificationPresetTrain() # Image transformation class
    
    model.train()
    print('Start training...')
