import gc
import random
import cv2 as cv
import pandas as pd
import tqdm as tqdm
from os import listdir

import torch
import torch.nn as nn

from utils import ClassificationPresetTrain, accuracy, eval_auc

def validation(model, criterion, bag_size, test_path, transform):
    model.eval()
    softmax = nn.Softmax(dim = 1)
    print("Prediction...")

def engine(model, device, criterion, optimizer, lr_scheduler, scaler, total_data, iterate, batch_size, bag_size, train_path, test_path):
    '''
    This function is written for training the binary classification MIL model
    Several modifications are needed for multiclass classification 
    (e.g., line 33 ~ 34 & line 50 ~ 58)
    '''
    roots = [f'{train_path}/{f}' for f in listdir(train_path)]
    data = [pd.read_csv(f'{root}.csv') for root in roots]
    images = [img_id['image_id'].tolist() for img_id in data]  
    patches = [num_patch[' patches'].tolist() for num_patch in data]   

    '''
    We are currently using 3 metrics for training evaluation: 
    training loss, accuarcy, and ROC_AUC
    '''
    counter = [0, 0] # Counters to iterate over the image dataset
    draw_order = [0, 1] # The draw order of class in each mini-batch
    labels, preds = [], [] # Place holder to store data to calculate ROC_AUC
    epoch_loss, epoch_err = 0.0, 0.0 # Evaluation metrics for training process
    epoch_size = total_data // batch_size 
    del data

    transform = ClassificationPresetTrain()
    
    model.train()
    print('Start training...')

    for i in tqdm(range(epoch_size * iterate)):
            
            '''Shuffles the draw_order to randomily select a class'''
            random.shuffle(draw_order) 
            for j in range(batch_size):
                temp = draw_order[j]
                row = counter[temp] % len(images[temp])
                counter[temp] += 1
                img_id = images[temp][row]
                max_num = patches[temp][row]
                
                '''Label for each bag'''
                empty = [0.0] * 2
                empty[temp] = 1.0
                
                '''Instances selection'''
                num_instance = bag_size if max_num > bag_size else max_num

                '''Stacking all images in each bag using batch'''
                for k in range(num_instance):
                    image = cv.imread(f'{roots[temp]}/{img_id}_{k:03d}.jpg')
                    try:
                        image = transform(image).unsqueeze(0)
                    except:
                        print(f'{roots[temp]}/{img_id}_{k}.jpg')
                    if k == 0:
                        img_stack = image
                    else:
                        img_stack = torch.cat([img_stack, image], dim=0)
                    del image
                    gc.collect()
                
                img_stack = img_stack.to(device).float()
                label = torch.tensor([empty]).to(device).float()
                output = model(img_stack)
                
                '''Gradient Normalization before accumulation'''
                loss = criterion(output, label) 
                loss = loss / batch_size

                epoch_loss += loss.item()
                error, pred = accuracy(output.cpu(), label.cpu())
                epoch_err += error
                labels.append(empty)
                preds.append(pred)
                
                del output, img_stack

                '''Accumulates scaled gradients'''
                scaler.scale(loss).backward()
                torch.cuda.empty_cache()

                '''Update the model using cumulated loss after each batch_size'''
                if j + 1 == batch_size:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    del loss
                    
            '''
            After defining the number of iteration (e.g., 80% of the total dataset), 
            we evaluate the current performance and save the model
            '''
            if (i != 0 and i % (epoch_size * 4 // 5) == 0) or i == epoch_size * iterate - 1:
                lr_scheduler.step()
                epoch_loss = epoch_loss / (epoch_size * 4 // 5)
                epoch_acc = 1 - epoch_err / (epoch_size * 4 // 5) / batch_size
                roc_auc = eval_auc(labels, preds)
                
                '''Print the status and reset the training loss, accuracy, and AUC'''
                print(f'Loss: {epoch_loss:.4f} | ACC: {epoch_acc:.4f} | AUC: {roc_auc:.4f}')
                epoch_loss, epoch_err = 0.0, 0.0
                labels, preds = [], []
                
                '''
                ToDo: Change the evaluation function to validate the perforamce using the rest of the dataset
                Currently, implementating this in the evaluation function causes CUDA out of memory (16GB GPU)
                '''
                validation(model, criterion, bag_size, test_path, transform)
                
                '''Save the checkpoint and continue training'''
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                            'lr_scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': i,}
                print('Saving the weights...')
                torch.save(checkpoint, f'./{i}_model.pth')
                model.train()
                
                '''Shuffles to the list of each class, which is similar to shuffle = True in PyTorch Dataloader'''
                for idx in range(len(roots)):
                    temp = list(zip(images[idx], patches[idx]))
                    random.shuffle(temp)
                    res1, res2 = zip(*temp)
                    images[idx], patches[idx] = list(res1), list(res2)
                del temp, res1, res2