import argparse

import torch
import torch.nn as nn

from model import MIL_EFF
from engine import engine

parser = argparse.ArgumentParser(
    description='MIL_EFF Training Script')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--epoch', default=None, type=int,
                    help='Epoch for training')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--bag_size', default=49, type=int,
                    help='Bag size for training')
parser.add_argument('--num_data', default=None, type=int,
                    help='Total number of image data for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--checkpoint_path', default=None, type=bool,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
args = parser.parse_args()

def main():
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    model = MIL_EFF()
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.11)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-5) 
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.33, total_iters=10)
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epoch-10, T_mult=2)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[10])
    scaler = torch.cuda.amp.GradScaler()
    start_iter = 0

    '''Load pretrained model if args.resume exists'''
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.param_groups[0]['capturable'] = True
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_iter = checkpoint["epoch"]
        scaler.load_state_dict(checkpoint["scaler"])

        del checkpoint
        torch.cuda.empty_cache()

    engine(model, device, criterion, optimizer, lr_scheduler, args.epoch - start_iter, args.batch_size, args.bag_size, args.num_data, scaler)

if __name__ == "__main__":
    main()
