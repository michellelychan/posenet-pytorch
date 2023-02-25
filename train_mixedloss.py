#create a train module that trains the model
# Path: posenet-pytorch/train.py
#inspiration: https://github.com/youngguncho/PoseNet-Pytorch/blob/master/posenet_simple.py 
#reference: https://github.com/Lornatang/MobileNetV1-PyTorch/blob/main/train.py

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
parser.add_argument('--train_image_dir', type=str, default='./images_train')
parser.add_argument('--test_image_dir', type=str, default= "./images_test")
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

class MixedLoss(nn.Module):
    '''
    ref: https://github.com/mks0601/PoseFix_RELEASE/blob/master/main/model.py
    input: {
        'heatmap': (N, C, X, Y, Z), unnormalized
        'coord': (N, C, 3)
    }
    target: {
        'heatmap': (N, C, X, Y, Z), normalized
        'coord': (N, C, 3)
    }
    '''
    def __init__(self, heatmap_weight=0.5):
    # def __init__(self, heatmap_weight=0.05):
        super(MixedLoss, self).__init__()
        self.w1 = heatmap_weight
        self.w2 = 1 - self.w1
        self.cross_entropy_loss = SoftmaxCrossEntropyWithLogits()

    def forward(self, input, target):
        pred_heatmap = input[0]
        pred_coord = input[1]
        gt_heatmap = target[0]
        gt_coord = target[1]
        # pred_heatmap, pred_coord = input['heatmap'], input['coord']
        # gt_heatmap, gt_coord = target['heatmap'], target['coord']

        # Heatmap loss
        N, C = pred_heatmap.shape[0:2]
        pred_heatmap = pred_heatmap.view(N*C, -1)
        gt_heatmap = gt_heatmap.view(N*C, -1)

        # Note, averaged over N*C
        hm_loss = self.cross_entropy_loss(pred_heatmap, gt_heatmap)

        # Coord L1 loss
        l1_loss = torch.mean(torch.abs(pred_coord - gt_coord))

        return self.w1 * hm_loss + self.w2 * l1_loss

class SoftmaxCrossEntropyWithLogits(nn.Module):
    '''
    Similar to tensorflow's tf.nn.softmax_cross_entropy_with_logits
    ref: https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
    The 'input' is unnormalized scores.
    The 'target' is a probability distribution.
    Shape:
        Input: (N, C), batch size N, with C classes
        Target: (N, C), batch size N, with C classes
    '''
    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()

    def forward(self, input, target):
        loss = torch.sum(-target * F.log_softmax(input, -1), -1)
        mean_loss = torch.mean(loss)
        return mean_loss

class PosenetDatasetImage(Dataset):
    def __init__(self, file_path, scale_factor=1.0, output_stride=16, train=True):
        self.file_path = file_path
        self.scale_factor = scale_factor
        self.output_stride = output_stride
        self.filenames = os.listdir(file_path)
        self.train = train

        
        self.data = [f.path for f in os.scandir(file_path) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
        self.filenames = [os.path.basename(file_path) for file_path in self.data]

        
        if  self.train:
            self.transforms = transforms.Compose([
                #not mandatory - at first don't apply augmentation first before applying 
                #transforms.RandomResizedCrop(256),
                #transforms.RandomHorizontalFlip(),

                #mandatory 
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                #mean and std values based on the pretrained model
                #mean value of the pixels of each channel [r, g, b]
                #std value of the pixels of each channel [r, g, b]
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # print("get_item: ", filename)
        input_image, draw_image, output_scale = posenet.read_imgfile(
            os.path.join(self.file_path, filename),
            scale_factor=self.scale_factor,
            output_stride=self.output_stride
        )
        
        
        # print(filename)
        # print(input_image.shape)
        
        input_image_tensor = torch.Tensor(input_image).cuda()
        #print("Tensor shape: ", input_image_tensor.shape[-2:])
        if input_image_tensor.shape[-2:] != (513, 513):
            input_image_resized = nn.functional.interpolate(input_image_tensor, size=(513, 513), mode='bilinear', align_corners=True)
            # print(f"Resized image {filename}: ", input_image_resized.shape)
            return input_image_resized, draw_image, output_scale
        else:
            return input_image_tensor, draw_image, output_scale
        
        # input_image = self.transforms(input_image)
        #return input_image, draw_image, output_scale
        

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # Set model to train mode
        model.train()
        
        print(train_loader)
        
        torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, target, _) in enumerate(train_loader):
            # print("ENUMERATE")
            
            data.cuda()
            data_squeezed = data.squeeze()
            target.cuda()
            
            # print("Train Target type: ", type(target))
            # print("Train Target [1] shape: ", target[1].shape)
            
            # data = torch.transpose(data, 1, 3)
            # data, target = data.cuda(), target.cuda()
            # Forward pass
            output = model(data_squeezed)
            # print("Output type: ", type(output))
            
            
            
            # print("finished model")
            # print("Train Output type: ", type(output[0]))
            # print("Train Output Length: ", len(output))
            
            #heatmap tensor = output[0] 
            #heatmap size is num of images x 17 keypoints x resolution x resolution 
            #eg. if image size is 225 with output stride of 16, then resolution is 15 
            # print(output[0].shape)
            # print(output[1].shape)
            # print(output[2].shape)
            # print(output[3].shape)
            
                        
            
            #get keypoint coordinates from output 
            # keypoint_coords = posenet.decode.decode_pose(output[0], output_scale=1.0)
            # print("Keypoint coords: ", keypoint_coords)
            
            # print(output)
            
            
            loss = criterion(output[0], target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target, _ in test_loader:
                data.cuda()
                data_squeezed = data.squeeze()
                target.cuda()
                # data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
                output = model(data_squeezed)
                test_loss += criterion(output[0], target).item()
        test_loss /= len(test_loader.dataset)

        print('Epoch: {} \tTrain Loss: {:.6f} \tTest Loss: {:.6f}'.format(
            epoch+1, loss.item(), test_loss))
        
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
    criterion = MixedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    train_dataset = PosenetDatasetImage(args.train_image_dir, train=True)
    test_dataset = PosenetDatasetImage(args.test_image_dir, train=False)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    # for x in train_dataset:
        # print("SHAPE+_+ ", x[1].shape)
        
        
    # print(next(iter(train_loader)))

        
    #image = next(iter(train_dataset))
    
    #image_label = next(iter(train_dataset))

    #print("image")
    #print(image.size())
    
    # print("image label")
    # print(image_label.size())
    
    
    # print(type(train_loader))
    # attrs = dir(train_loader)
    # for attr in attrs:
    #     print(f"{attr}: {type(getattr(train_loader, attr))}")

    # Training loop
    train(model, train_loader, test_loader, criterion, optimizer, num_epochs)
    print('Setting up...')
    

    

if __name__ == "__main__":
    main()