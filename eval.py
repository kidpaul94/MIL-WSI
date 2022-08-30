import argparse

import torch

from model import MIL_EFF
from engine import validation

parser = argparse.ArgumentParser(
    description='MIL_EFF Evaluation Script')
parser.add_argument('--trained_model',
                    default=None, type=str,
                    help='Trained state_dict file path to open.')
parser.add_argument('--test_path', default=None, type=str,
                    help='Path of train dataset')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--bag_size', default=49, type=int,
                    help='Bag size for training')
parser.add_argument('--num_data', default=None, type=int,
                    help='Total number of image data for training')
args = parser.parse_args()

def main():
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

    print('Loading the model...')
    model = MIL_EFF()
    checkpoint = torch.load('../input/temp-weights/2086_model.pth')
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    model.to(device)
    print('Model is loaded.')

    validation(model, device, args.test_path, args.bag_size)

if __name__ == '__main__':
    main()