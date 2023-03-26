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
from visualizers import *
from scipy.optimize import linear_sum_assignment


CUDA_LAUNCH_BLOCKING=1

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--train_image_dir', type=str, default='./images_train')
parser.add_argument('--test_image_dir', type=str, default= "./images_test")
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--scale_factor', type=float, default=1.0)

args = parser.parse_args()

#Loss function with Hough Voting 

class MultiPersonHeatmapOffsetAggregationLoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(MultiPersonHeatmapOffsetAggregationLoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        self.smoothl1loss = nn.SmoothL1Loss(reduction='none')


        self.use_target_weight = use_target_weight

    def forward(self, score_threshold, pred_keypoints, target_keypoints, pred_heatmaps, target_heatmaps, pred_offsets, target_offsets):
        """
        Compute the heatmap offset aggregation loss with Hough voting
        Reference from paper Towards Accurate Multi-person Pose Estimation in the Wild
        :param pred_heatmaps: predicted heatmaps of shape 
        :param target_heatmaps: target heatmaps of shape (num_joints, height, width)
        :param pred_offsets: predicted offsets of shape (num_joints*2, height, width)
        :param target_offsets: target offsets of shape (num_joints*2, height, width)
        :return: loss value
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        pred_heatmaps_binarized = torch.where(pred_heatmaps > score_threshold, torch.ones_like(pred_heatmaps), torch.zeros_like(pred_heatmaps))
        target_heatmaps_binarized = torch.where(target_heatmaps > score_threshold, torch.ones_like(target_heatmaps), torch.zeros_like(target_heatmaps))
        
        heatmap_loss = self.bceloss(pred_heatmaps_binarized, target_heatmaps_binarized)
        
        print("pred_heatmaps.requires_grad:", pred_heatmaps.requires_grad)
        print("pred_offsets.requires_grad:", pred_offsets.requires_grad)
        
        
        # Print the tensors to check their gradient status
        #find the euclidean distance between the predicted and target offset vectors
        #euclidean distance = sqrt((pred_x - target_x)^2 + (pred_y - target_y)^2)
        # distances = torch.sqrt(torch.pow(pred_x - target_x, 2) + torch.pow(pred_y - target_y, 2))
        #?? should I normalize the distances? 
        
        
        #compute the difference between the predicted offsets and the ground truth offsets
        diff = pred_offsets - target_offsets
        print("pred keypoints shape: ", pred_keypoints.shape)
        print("target keypoints shape: ", target_keypoints.shape)
        
        #Compute the distance between the keypoint position lk and the position xi
        pred_keypoints = torch.from_numpy(pred_keypoints)
        target_keypoints = torch.from_numpy(target_keypoints)
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
    def __init__(self, file_path, ground_truth_keypoints_dir=None, scale_factor=1.0, output_stride=16, train=True):
        self.file_path = file_path
        self.scale_factor = scale_factor
        self.output_stride = output_stride
        self.filenames = os.listdir(file_path)
        self.train = train
        self.ground_truth_keypoints_dir = ground_truth_keypoints_dir
        
        if ground_truth_keypoints_dir:
            image_file_names = [os.path.splitext(file)[0] for file in self.filenames if file.endswith((".jpg", ".png"))]
            self.keypoints, self.heatmaps, self.offset_vectors = load_ground_truth_data(image_file_names, self.ground_truth_keypoints_dir)
            self.is_ground_truth = True
            print("PosenetDatasetImage filenames: ", self.filenames)
        else:
            self.is_ground_truth = False
        
        
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
        print("____getitem____ idx: ", idx)

        filename = self.filenames[idx]
        print("____getitem____ filename: ", filename)
        
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
        
        if self.is_ground_truth:
            print("print length of keypoints: ", len(self.keypoints))
            keypoints = self.keypoints[idx]

            heatmaps = self.heatmaps[idx]
            offset_vectors = self.offset_vectors[idx]
            
            return input_image_tensor, draw_image, output_scale, filename, keypoints, heatmaps, offset_vectors
                
        else:
            return input_image_tensor, draw_image, output_scale, filename

        
        # input_image = self.transforms(input_image)
        #return input_image, draw_image, output_scale
        
        
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, output_stride, train_image_path, test_image_path, output_dir, scale_factor, is_train=True):
    for epoch in range(num_epochs):
        
        score_threshold = 0.25
        
        # Set model to train mode
        if is_train:
            model.train()
              
            print(train_loader)
        
            print("train loader: ", next(iter(train_loader)))

            for batch_idx, (data, draw_image, output_scale, filenames, _, _, _) in enumerate(train_loader):
                # print("ENUMERATE")
            
                print("batch size: ", train_loader.batch_size)
            
                data.cuda()
                print("data shape: ", data.shape)
    
                data_squeezed = data.squeeze()
        
                print("data_squeezed shape: ", data_squeezed.shape)
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

                    #turn epoch to text
                    appended_text = "train_" + str(epoch) + "_"
                
                    is_train = True
                    
                    #decoder with single pose 
                    
                    instance_keypoint_coords, instance_keypoint_scores , train_heatmaps, train_offsets = decode_pose_from_batch_item(epoch, train_image_path, filenames[item_idx], item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)
                    draw_coordinates_to_image_file(appended_text, train_image_path, output_dir, output_stride, scale_factor, instance_keypoint_scores, instance_keypoint_coords, filenames[item_idx], include_displacements=False)

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
            for batch_idx, (data, draw_image, output_scale, filenames) in enumerate(test_loader):
                data.cuda()
                data_squeezed = data.squeeze()
                # data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
                output = model(data_squeezed)
                
                test_loss = 0
                
                
                #iterate through the batch size
                for item_idx, item in enumerate(output[0]):
                    
                    is_train_decoding = False
                    
                    offsets = output[1][item_idx]
                    displacements_fwd = output[2][item_idx]
                    displacements_bwd = output[3][item_idx]
                    
                    test_heatmaps = item
                    height = test_heatmaps.shape[1]
                    width = test_heatmaps.shape[2]
                    
                    print("test_heatmap shape: ", test_heatmaps.shape)
                    print("test displacement fwd shape: ", displacements_fwd.shape)
                    
                    pose_scores, keypoint_scores, keypoint_coords = decode_pose_from_batch_item(epoch, test_image_path, filenames[item_idx], item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train_decoding)

                    print("pose_scores shape: ", pose_scores.shape)
                    print("offsets shape: ", offsets.shape)
                    print("keypoint_scores shape: ", keypoint_scores.shape)
                    print("keypoint_coords shape: ", keypoint_coords.shape)
                    
                    appended_text = "test_"
                    

                    draw_coordinates_to_image_file(appended_text, test_image_path, output_dir, output_stride, scale_factor, pose_scores,keypoint_scores, keypoint_coords, filenames[item_idx], include_displacements=False)

                    ######## 
                    test_loss += criterion(score_threshold, keypoint_coords, keypoint_coords, test_heatmaps, test_heatmaps, offsets, offsets).item()

            test_loss /= len(test_loader.dataset)

            # print('Epoch: {} \tTrain Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch+1, test_loss))


def decode_pose_from_batch_item(epoch, image_path, filename, item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train):
    heatmaps = item
    
    ######
    print(is_train, " train offsets: ", offsets.shape)
    print(is_train, " train displacements_fwd shape: ", displacements_fwd.shape)
    #find the root keypoint id's coordinates            
    #sorted scores vectors and location of the max of heatmap? 

    ###########
    #decode single pose 
    ###########
    #highest_scores, highest_score_coords  = posenet.decode_multi.build_part_with_score_torch_single_pose(score_threshold, LOCAL_MAXIMUM_RADIUS, heatmaps)
    #root_score, root_id, root_image_coord = posenet.decode.find_root(highest_scores, highest_score_coords)
    
    #highest_score_coords = highest_score_coords * scale_factor * output_stride

    #save coordinates into image before decoding pose
    #appended_text = "before_decode_" + str(epoch) + "_"
    
    
    # print(is_train, " train displacements_fwd reshaped shape: ", displacements_fwd_reshaped.shape) 
                


    # offsets_reshaped = offsets.detach().cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    heatmaps = torch.tensor(heatmaps, requires_grad=is_train)

    # print("displacements shape before draw_doordinates_to_image_file: ", displacements_fwd.shape)
    # print("displacements shape after draw_doordinates_to_image_file: ", displacements_fwd_reshaped.shape)

    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps,
                offsets,
                displacements_fwd,
                displacements_bwd,
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=score_threshold)

    
    
    
    # instance_keypoint_scores, instance_keypoint_coords, displacement_vectors = posenet.decode.decode_pose(root_score, root_id, root_image_coord, heatmaps, offsets_reshaped, output_stride, displacements_fwd_reshaped, displacements_bwd_reshaped)
    
    appended_text = "after_decode_"
    output_dir = "output_after_decode"

    # displacements_fwd_reshaped = displacements_fwd.detach().cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    # displacements_bwd_reshaped = displacements_bwd.detach().cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))

    draw_coordinates_to_image_file(appended_text, image_path, output_dir, output_stride, scale_factor, pose_scores, keypoint_scores, keypoint_coords, filename, displacements_fwd, displacements_bwd, include_displacements=True)

    # instance_keypoint_coords = torch.tensor(instance_keypoint_coords, requires_grad=is_train)
    # instance_keypoint_coords.cuda()
    # offsets = torch.tensor(offsets, requires_grad=is_train)
                    
    return pose_scores, keypoint_scores, keypoint_coords
            

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
    criterion = MultiPersonHeatmapOffsetAggregationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    

    # Training loop
    train_image_path = args.train_image_dir
    test_image_path = args.test_image_dir
    output_dir = args.output_dir
    scale_factor = args.scale_factor
    ground_truth_keypoints_dir = "./keypoints_updated"
    
    is_train = False
    
    train_dataset = PosenetDatasetImage(train_image_path, ground_truth_keypoints_dir, scale_factor=1.0, output_stride=output_stride, train=True)
    test_dataset = PosenetDatasetImage(test_image_path, scale_factor=1.0, output_stride=output_stride, train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train(model, train_loader, test_loader, criterion, optimizer, num_epochs, output_stride, train_image_path, test_image_path, output_dir, scale_factor, is_train)

    print('Setting up...')


if __name__ == "__main__":
    main()