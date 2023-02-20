#create a train module that trains the model
# Path: posenet-pytorch/train.py
#inspiration: https://github.com/youngguncho/PoseNet-Pytorch/blob/master/posenet_simple.py 

#install torch with pip
# Path: posenet-pytorch/train.py

import cv2
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import posenet
import time
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

class PosenetDatasetImage(Dataset):
    def __init__(self, file_path, scale_factor=1.0, output_stride=16, train=True):
        self.file_path = file_path
        self.scale_factor = scale_factor
        self.output_stride = output_stride
        self.filenames = os.listdir(file_path)
        self.train = train

        # Load data from file_path
        # e.g., using pandas or numpy
        # self.data = ... 
        self.data = [f.path for f in os.scandir(file_path) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        if  self.train:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self .filenames[idx]
        input_image, draw_image, output_scale = posenet.read_imgfile(
            os.path.join(self.file_path, filename),
            scale_factor=self.scale_factor,
            output_stride=self.output_stride
        )
        # x = sample[:-1]
        # y = sample[-1]
        return input_image, draw_image, output_scale

def main():

    #instatiate model 
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    # Set up training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    train_dataset = PosenetDatasetImage(args.image_dir, train=True)
    test_dataset = PosenetDatasetImage(args.image_dir, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    for epoch in range(num_epochs):
        pass
    print('Setting up...')

if __name__ == "__main__":
    main()