#create a train module that trains the model
# Path: posenet-pytorch/train.py
#inspiration: https://github.com/youngguncho/PoseNet-Pytorch/blob/master/posenet_simple.py 
#reference: https://github.com/Lornatang/MobileNetV1-PyTorch/blob/main/train.py

#install torch with pip
# Path: posenet-pytorch/train.py

#resolution (image size: 225; stride: 16) 
#// 15 = ((225 - 1) / 16) + 1
#output[0]: heatmap  [15, 17, 33, 33] 
#output[1]: offset vectors [15, 34, 33, 33] 
#output[2]: displacement forward [15, 32, 33, 33] 
#output[3]: displacement backward [15, 32, 33, 33]

import cv2
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import posenet
import time
from torchvision import transforms
import matplotlib.pyplot as plt
from ground_truth import *
from posenet.decode_multi import *


CUDA_LAUNCH_BLOCKING=1

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--train_image_dir', type=str, default='./images_train')
parser.add_argument('--test_image_dir', type=str, default= "./images_test")
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

#Loss function with Hough Voting 
class HeatmapOffsetAggregationLoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(HeatmapOffsetAggregationLoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        self.smoothl1loss = nn.SmoothL1Loss(reduction='none')


        self.use_target_weight = use_target_weight

    def forward(self, score_threshold, pred_keypoints, target_keypoints, pred_heatmaps, target_heatmaps, pred_offsets, target_offsets):
        """
        Compute the heatmap offset aggregation loss with Hough voting
        Reference from paper Towards Accurate Multi-person Pose Estimation in the Wild
        :param pred_heatmaps: predicted heatmaps of shape (batch_size, num_joints, height, width)
        :param target_heatmaps: target heatmaps of shape (batch_size, num_joints, height, width)
        :param pred_offsets: predicted offsets of shape (batch_size, 2, height, width)
        :param target_offsets: target offsets of shape (batch_size, 2, height, width)
        :return: loss value
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        pred_heatmaps_binarized = torch.where(pred_heatmaps > score_threshold, torch.ones_like(pred_heatmaps), torch.zeros_like(pred_heatmaps))
        target_heatmaps_binarized = torch.where(target_heatmaps > score_threshold, torch.ones_like(target_heatmaps), torch.zeros_like(target_heatmaps))
        
        heatmap_loss = self.bceloss(pred_heatmaps_binarized, target_heatmaps_binarized)
        
        print("pred_heatmaps.requires_grad:", pred_heatmaps.requires_grad)
        print("pred_offsets.requires_grad:", pred_offsets.requires_grad)
        
        
        # Print the tensors to check their gradient status

        
        #convert pred_heatmaps and target_heatmaps into a binary heatmap around radius R around predicted and target keypoints
        #R = 1.5
        #radius R around predicted and target keypoints should be 1, the rest should be 0
        # pred_x = torch.index_select(pred_offsets, 1, torch.arange(0, 17).to(device))
        # pred_y = torch.index_select(pred_offsets, 1, torch.arange(17, 34).to(device))
        # target_x = torch.index_select(target_offsets, 1, torch.arange(0, 17).to(device))
        # target_y = torch.index_select(target_offsets, 1, torch.arange(17, 34).to(device))
        
        #find the euclidean distance between the predicted and target offset vectors
        #euclidean distance = sqrt((pred_x - target_x)^2 + (pred_y - target_y)^2)
        # distances = torch.sqrt(torch.pow(pred_x - target_x, 2) + torch.pow(pred_y - target_y, 2))
        #?? should I normalize the distances? 
        
        
        #compute the difference between the predicted offsets and the ground truth offsets
        diff = pred_offsets - target_offsets
        print("pred keypoints shape: ", pred_keypoints.shape)
        print("target keypoints shape: ", target_keypoints.shape)
        
        #Compute the distance between the keypoint position lk and the position xi
        distances = torch.norm(pred_keypoints - target_keypoints, dim=1)
        print("distance shape: ", distances.shape)



        zero_distances = torch.zeros_like(distances)

        offset_loss = self.smoothl1loss(distances, zero_distances).mean()

        print("heatmap loss: ", heatmap_loss)

        #print the value of the offset loss
        print("offset loss: ", offset_loss)

        loss = 4 * heatmap_loss + offset_loss
        print("loss: ", loss)
        return loss


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
        

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, output_stride):
    
    for epoch in range(num_epochs):
        # Set model to train mode
        
            
        model.train()
        
        
              
        print(train_loader)
        
        print("train loader: ", next(iter(train_loader)))
        
        score_threshold = 0.5

        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            # print("ENUMERATE")
            

            print("batch size: ", train_loader.batch_size)
            
            data.cuda()
            print("data shape: ", data.shape)
    
            data_squeezed = data.squeeze()
        
            print("data_squeezed shape: ", data_squeezed.shape)
            target.cuda()
            output = model(data_squeezed)
            
            #heatmap tensor = output[0] 
            #heatmap size is num of images x 17 keypoints x resolution x resolution 
            #eg. if image size is 225 with output stride of 16, then resolution is 15 
            
        
            #get the heatmaps batch from the heatmaps in output [0] according to batch idx
            print("batch_idx: ", batch_idx)
            print("output[0] shape: ", output[0].shape)
            
            #iterate through the batch size
            for item_idx, item in enumerate(output[0]):
                train_heatmaps = item
                height = train_heatmaps.shape[1]
                width = train_heatmaps.shape[2]
                
                offsets = output[1][item_idx]
                displacements_fwd = output[2][item_idx]
                displacements_bwd = output[2][item_idx]
                
                is_train = True
                
                instance_keypoint_coords, _ , train_heatmaps, train_offsets = decode_pose_from_batch_item(item, offsets, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)

                loss = criterion(score_threshold, instance_keypoint_coords, instance_keypoint_coords, train_heatmaps, train_heatmaps, train_offsets, train_offsets)
            
                print("loss.requires_grad:", loss.requires_grad)
                print("LOSS: ", loss)
            
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('[Train] Epoch [{}/{}], Batch [{}/{}], Item [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, batch_idx+1, len(train_loader), item_idx+1, output[0].shape[0], loss.item()))

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

                
                #iterate through the batch size
                for item_idx, item in enumerate(output[0]):
                    is_train = False
                    offsets = output[1][item_idx]
                    displacements_fwd = output[2][item_idx]
                    displacements_bwd = output[3][item_idx]
                    
                    instance_keypoint_coords, _ , test_heatmaps, test_offsets = decode_pose_from_batch_item(item, offsets, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)

                    ######## 
                    test_loss += criterion(score_threshold, instance_keypoint_coords, instance_keypoint_coords, test_heatmaps, test_heatmaps, test_offsets, test_offsets).item()
                
        test_loss /= len(test_loader.dataset)

        print('Epoch: {} \tTrain Loss: {:.6f} \tTest Loss: {:.6f}'.format(
            epoch+1, loss.item(), test_loss))
        
    heatmap = output[0]
    print_heatmap(heatmap)

def decode_pose_from_batch_item(item, offsets, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train):
    heatmaps = item
    
    ######
    print(is_train, " train offsets: ", offsets.shape)
    print(is_train, " train displacements_fwd shape: ", displacements_fwd.shape)
    #find the root keypoint id's coordinates            
    #sorted scores vectors and location of the max of heatmap? 
            
    highest_scores, highest_score_coords  = posenet.decode_multi.build_part_with_score_torch_single_pose(score_threshold, LOCAL_MAXIMUM_RADIUS, heatmaps)
    root_score, root_id, root_image_coord = posenet.decode.find_root(highest_scores, highest_score_coords)
                
    displacements_fwd_reshaped = displacements_fwd.detach().cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    print(is_train, " train displacements_fwd reshaped shape: ", displacements_fwd_reshaped.shape) 
                
    displacements_bwd_reshaped = displacements_bwd.detach().cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    offsets_reshaped = offsets.detach().cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    heatmaps = torch.tensor(heatmaps, requires_grad=is_train)

    # train_offsets_reshaped_tensor = torch.tensor(train_offsets_reshaped, requires_grad=True)
            
    #decode pose 
    print(is_train, " train root_id: ", root_id)
    print(is_train, " train root_score: ", root_score)
    print(is_train, " train root_image_coord: ", root_image_coord)
    instance_keypoint_scores, instance_keypoint_coords = posenet.decode.decode_pose(root_score, root_id, root_image_coord, heatmaps, offsets_reshaped, output_stride, displacements_fwd_reshaped, displacements_bwd_reshaped)
                    
    instance_keypoint_coords = torch.tensor(instance_keypoint_coords, requires_grad=is_train)
    instance_keypoint_coords.cuda()
    offsets = torch.tensor(offsets, requires_grad=is_train)
                    
    print(is_train, " train instance_keypoint_coords.requires_grad: ", instance_keypoint_coords.requires_grad)
    print(is_train, " train offsets.requires_grad: ", offsets.requires_grad)
    print(is_train, " train heatmaps.requires_grad: ", heatmaps.requires_grad)

    return instance_keypoint_coords, instance_keypoint_scores, heatmaps, offsets
            

def print_heatmap(heatmap):
    #print heatmap for each image
    os.makedirs('heatmaps', exist_ok=True)
    
    #loop through the 15 images 
    for i in range(heatmap.shape[0]):
        # Create a new directory for this image
        os.makedirs(f'heatmaps/image_{i}', exist_ok=True)
        
        #loop through each joint 
        for j in range(heatmap.shape[1]):
            joint_heatmap = heatmap[i, j, :, :].squeeze()
            
            # Plot the heatmap
            plt.gca().set_aspect('equal', adjustable='box')
            plt.imshow(joint_heatmap.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            
            # Save the heatmap in the corresponding image folder
            plt.savefig(f'./heatmaps/image_{i}/joint_{j}_heatmap.png')
            
            # Clear the plot
            plt.clf()

def main():

    #instatiate model 
    model = posenet.load_model(args.model)
    model = model.cuda()
    
    for param in model.parameters():
        param.requires_grad = True
    
    output_stride = model.output_stride

    # Set up training parameters
    batch_size = 2
    learning_rate = 0.001
    num_epochs = 10
    
    plt = points_to_heatmap(4.5, 4.7, 21)
    
    # Define loss function and optimizer
    criterion = HeatmapOffsetAggregationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    train_dataset = PosenetDatasetImage(args.train_image_dir, train=True)
    test_dataset = PosenetDatasetImage(args.test_image_dir, train=False)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    train(model, train_loader, test_loader, criterion, optimizer, num_epochs, output_stride)

    print('Setting up...')
    


if __name__ == "__main__":
    main()