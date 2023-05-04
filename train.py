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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import posenet
import time
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from ground_truth_dataloop import *
from posenet.decode_multi import *
from visualizers import *
from scipy.optimize import linear_sum_assignment
import wandb
import torch.optim as optim

os.environ["WANDB_NOTEBOOK_NAME"] = "./train_model_wandb.ipynb"



CUDA_LAUNCH_BLOCKING=1

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--train_image_dir', type=str, default='./images_train')
parser.add_argument('--test_image_dir', type=str, default= "./images_train")
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--scale_factor', type=float, default=1.0)


args = parser.parse_args()

#Loss function with Hough Voting 

# class MaskedBCEWithLogitsLoss(nn.Module):
#     def __init__(self):
#         super(MaskedBCEWithLogitsLoss, self).__init__()
#         self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, input, target, mask):
#         # Compute BCEWithLogitsLoss
#         loss = self.bce_with_logits_loss(input, target)

#         # Apply the mask
#         masked_loss = loss * mask

#         # Compute the mean loss over the masked elements
#         # mean_loss = torch.sum(masked_loss) / torch.sum(mask)

#         return masked_loss

class MultiPersonHeatmapOffsetAggregationLoss(nn.Module):
    def __init__(self, radius=3, heatmap_weight=4.0, offset_weight=1.0, use_target_weight=False):
        super(MultiPersonHeatmapOffsetAggregationLoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        self.smoothl1loss = nn.SmoothL1Loss(reduction='none')
        self.radius = radius
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.use_target_weight = use_target_weight
        

    import torch.nn.functional as F

    def create_mask(self, ground_truth, threshold=0.1):
        # Threshold the ground truth heatmaps to create a binary mask
        mask = (ground_truth > threshold).float()

        # Apply dilation to create a disk-like region around each keypoint
        padding = self.radius
        kernel_size = 2 * self.radius + 1
        mask = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)

        mask = mask.cuda()
        
        return mask


    def create_binary_target_heatmap(self, target_heatmaps, target_keypoints, radius=3):
        #TODO: check if binary target heatmaps is in the right shape and if it should be zeros_like
        binary_target_heatmaps = torch.zeros_like(target_heatmaps)

#         print("target_heatmaps shape: ", target_heatmaps.shape)
#         print("target_keypoints shape: ", target_keypoints.shape)
              
        for k in range(target_keypoints.shape[0]):

            x, y = target_keypoints[k, 0], target_keypoints[k, 1]
            # print("x: ", x)
            # print("y: ", y)
            if (x != 0 and x != -1) or (y != 0 and y != -1):
                x, y = int(x.item()), int(y.item())
                y_min, y_max = max(0, y - radius), min(binary_target_heatmaps.shape[1], y + radius + 1)
                x_min, x_max = max(0, x - radius), min(binary_target_heatmaps.shape[2], x + radius + 1)

                y_indices, x_indices = np.mgrid[y_min:y_max, x_min:x_max]
                y_indices, x_indices = torch.tensor(y_indices), torch.tensor(x_indices)
                distances = torch.sqrt((y_indices - y)**2 + (x_indices - x)**2)

                binary_target_heatmaps[k, y_min:y_max, x_min:x_max] = (distances <= radius).float()

        return binary_target_heatmaps

    def forward(self, pred_heatmaps, target_heatmaps, target_keypoints, pred_offsets, target_offsets):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        binary_target_heatmaps = torch.zeros_like(target_heatmaps)
        
        # Heatmap loss
        max_num_poses = 15
        loss = 0.0
        
        #TODO update num_people logic
        num_people = 15
        heatmap_loss = torch.tensor(0.0)

        for pose in range(num_people):
            binary_target_heatmaps[pose, :, :, :] = self.create_binary_target_heatmap(target_heatmaps[pose], target_keypoints[pose], self.radius)
            pose_heatmap_loss = self.bceloss(pred_heatmaps, binary_target_heatmaps)

            print("pred_heatmaps shape: ", pred_heatmaps.shape)
            print("binary_target_heatmaps shape: ", binary_target_heatmaps.shape)
            print("target_heatmaps shape: ", target_heatmaps.shape)
            
            heatmap_loss += pose_heatmap_loss
        
        heatmap_loss /= num_people
        
        
        # Offset loss
        mask = self.create_mask(target_heatmaps[pose])
        mask = mask.unsqueeze(-1)
        
        go_forward = False
        offset_loss = torch.tensor(0.02)
        
        if go_forward: 
            pred_offsets = pred_offsets.view(1, 17, 2, 33, 33).permute(0, 1, 3, 4, 2)
            print("pred_offsets shape: ", pred_offsets.shape)
                
            #turn ground truth offsets from shape [17,2] to shape [17, 33, 33]
            ground_truth_offset_maps = create_ground_truth_offset_maps(target_keypoints, height=33, width=33)
            print("ground_truth_offset_maps shape: ", ground_truth_offset_maps.shape)
    
            print("mask shape: ", mask.shape)
            print("pred_offsets shape: ", pred_offsets.shape)
            print("ground_truth_offset_maps device: ", ground_truth_offset_maps.device)
            print("pred_offsets device: ", pred_offsets.device)
            print("mask device: ", mask.device)
            
            masked_true_offsets = ground_truth_offset_maps * mask
            
            masked_pred_offsets = pred_offsets * mask
            
            offset_loss += self.smoothl1loss(masked_pred_offsets, masked_true_offsets).mean()
            
    
            offset_loss = offset_loss / num_people
            
            aggregated_gt_heatmaps = self.aggregate_ground_truth_heatmaps(binary_target_heatmaps)
            print("aggregated_gt_heatmaps shape: ", aggregated_gt_heatmaps.shape)
            print("binary target_heatmaps shape: ", binary_target_heatmaps.shape)
            print("pred_heatmaps shape: ", pred_heatmaps.shape)
            
            print("heatmap loss: ", heatmap_loss)
            
            # print("pred offsets: ", pred_offsets)
            # print("target offsets: ", target_offsets)
            print("offset loss: ", offset_loss)

        loss += self.heatmap_weight * heatmap_loss + self.offset_weight * offset_loss
            
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
            print("--inside dataset class init --")
            print("keypoints shape: ", self.keypoints.shape)
            print("heatmaps shape: ", self.heatmaps.shape)
            print("offest_vectors shape: ", self.offset_vectors.shape)
            # self.keypoints = torch.Tensor(self.keypoints).cuda()
            # self.heatmaps = torch.Tensor(self.heatmaps).cuda()
            # self.offset_vectors = torch.Tensor(self.offset_vectors).cuda()
            
            
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
                
                transforms.Normalize(mean=[5.4476, 8.3573, 7.5377], std=[3.6566, 3.5510, 4.0362])
            ])
            
            if ground_truth_keypoints_dir:
                self.keypoints = torch.Tensor(self.keypoints).cuda().requires_grad_(False)
                self.heatmaps = torch.Tensor(self.heatmaps).cuda().requires_grad_(False)
                self.offset_vectors = torch.Tensor(self.offset_vectors).cuda().requires_grad_(False)
            
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[5.4476, 8.3573, 7.5377], std=[3.6566, 3.5510, 4.0362])
            ])


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print("____getitem____ idx: ", idx)

        filename = self.filenames[idx]
        # print("____getitem____ filename: ", filename)
        
        # print("get_item: ", filename)
        input_image, draw_image, output_scale = posenet.read_imgfile(
            os.path.join(self.file_path, filename),
            scale_factor=self.scale_factor,
            output_stride=self.output_stride
        )
        
        # print("----input image: ----")
        # print(input_image)
        
        # print(filename)
        # print(input_image.shape)
        
        input_image_tensor = torch.Tensor(input_image).cuda()
        
        #print("Tensor shape: ", input_image_tensor.shape[-2:])
        if input_image_tensor.shape[-2:] != (513, 513):
            input_image_resized = nn.functional.interpolate(input_image_tensor, size=(513, 513), mode='bilinear', align_corners=True)
            # print(f"Resized image {filename}: ", input_image_resized.shape)
        
        if self.is_ground_truth:
            # print("print length of keypoints: ", len(self.keypoints))
            keypoints = self.keypoints[idx]
            heatmaps = self.heatmaps[idx]
            offset_vectors = self.offset_vectors[idx]
            
            
            return input_image_tensor, draw_image, output_scale, filename, keypoints, heatmaps, offset_vectors
                
        else:
            return input_image_tensor, draw_image, output_scale, filename

def get_dataset_mean_std(dataset):
    # Calculate the mean and standard deviation for each channel
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    for i, (input_image_tensor, draw_image, _, _, _, _, _) in enumerate(dataset):
        # print("number of outputs of dataset: ", len(next(iter(dataset))))
        # print("draw_image type: ", type(draw_image))
        # print("draw_image shape: ", draw_image.shape)
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for i in range(3):
            mean[i] = draw_image[..., i].mean()
            std[i] = draw_image[..., i].std()

    mean /= len(dataset)
    std /= len(dataset)
    print(f'mean: {mean}')
    print(f'std: {std}')
    
    return mean, std

        
def create_ground_truth_offset_maps(ground_truth_keypoints, height, width, scale_factor=8):
    ground_truth_keypoints = ground_truth_keypoints.cuda()
    #TODO : find the num_pose for each ground truth and use that instead 
    num_poses = 15
    ground_truth_offset_maps = torch.zeros((num_poses, NUM_KEYPOINTS, height, width, 2), dtype=torch.float32).cuda()
    
    print("ground_truth_offset_maps shape: ", ground_truth_offset_maps.shape)
    print("ground_truth_keypoints shape: : ", ground_truth_keypoints.shape)
    
    for n in range(num_poses):
        for k in range(NUM_KEYPOINTS):
            for i in range(height):
                for j in range(width):
                    y_coord = i * scale_factor
                    x_coord = j * scale_factor
                    ground_truth_offset_maps[n, k, i, j] = ground_truth_keypoints[n,k] - torch.tensor([y_coord, x_coord]).cuda()
                    
    # reshaped_ground_truth_offset_maps = torch.cat([ground_truth_offset_maps[:, :, :, 0], ground_truth_offset_maps[:, :, :, 1]], dim=0)
    return ground_truth_offset_maps


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, output_stride, train_image_path, test_image_path, output_dir, scale_factor, is_train=True):
    step = 0
    score_threshold = 0.5
    train_num_batches = len(train_loader)

    for epoch in range(num_epochs):
        
        epoch_start_time = time.time()
        batch_checkpoint = 1
        
        epoch_durations = []
        running_loss_value = 0
        test_loss_value = 0
        test_loss = torch.zeros(1)
        
        # Set model to train mode
        # print("Initial model weights:")
        # for name, param in model.named_parameters():
        #     print(name, param.data)
        
        # print("Initial Model weight norms:")
        # for name, param in model.named_parameters():
        #     print(name, param.data.norm())

            
        if is_train:
            model.train()
              
            print(train_loader)
        
            # print("train loader: ", next(iter(train_loader)))

            for batch_idx, (data, draw_image, output_scale, filenames, ground_truth_keypoints, ground_truth_heatmaps, ground_truth_offsets) in enumerate(train_loader):
                # print("ENUMERATE")
            
                # print("batch size: ", train_loader.batch_size)
            
                data.cuda()
                # print("data shape: ", data.shape)
    
                data_squeezed = data.squeeze()
        
                # print("data_squeezed shape: ", data_squeezed.shape)
                output = model(data_squeezed)

                
                batch_loss = 0
                
                #heatmap tensor = output[0] 
                #heatmap size is num of images x 17 keypoints x resolution x resolution 
                #eg. if image size is 225 with output stride of 16, then resolution is 15 
                #iterate through the batch size
                for item_idx, item in enumerate(output[0]):
                    offsets = output[1][item_idx]
                    displacements_fwd = output[2][item_idx]
                    displacements_bwd = output[2][item_idx]

                    train_heatmaps = item
                    # print("item (heatmap) type: ", type(item))
                    
                    height = train_heatmaps.shape[1]
                    width = train_heatmaps.shape[2]
                    
                    # instance_keypoint_coords, instance_keypoint_scores , train_heatmaps, train_offsets = decode_pose_from_batch_item(epoch, train_image_path, filenames[item_idx], item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)
                    pose_scores, keypoint_scores, keypoint_coords, decoded_offsets = decode_pose_from_batch_item(epoch, train_image_path, filenames[item_idx], item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)
                    # print("---- keypoint_coords: ----")
                    # print(keypoint_coords)         
                    
                    #turn epoch to text
                    appended_text = "train_" + str(epoch) + "_"
                    print("pose_scores shape: ", pose_scores.shape)
                    print("keypoint_scores shape: ", keypoint_scores.shape)
                    draw_coordinates_to_image_file(appended_text, train_image_path, output_dir, output_stride, scale_factor, pose_scores, keypoint_scores, keypoint_coords, filenames[item_idx], include_displacements=False)

                    decoded_offsets = torch.from_numpy(decoded_offsets)
                    decoded_offsets = decoded_offsets.to('cuda')
                    
                    # print("decoded_offsets: ", decoded_offsets)
                    print("decoded_offsets shape: ", decoded_offsets.shape)

                    keypoint_coords = torch.from_numpy(keypoint_coords)
                    keypoint_coords = keypoint_coords.to('cuda')           
                    
                    print("offsets shape: ", offsets.shape)
                    
                    loss = criterion(train_heatmaps, ground_truth_heatmaps[item_idx] , ground_truth_keypoints[item_idx],  offsets, ground_truth_offsets[item_idx])

                    # Backward pass
                    optimizer.zero_grad()
                    
                    print("loss shape: ", loss.shape)

                    print("loss: ", loss)

                    print('[Train] Epoch [{}/{}], Batch [{}/{}], Item [{}/{}], Loss: {:.4f}'
                          .format(epoch+1, num_epochs, batch_idx+1, len(train_loader), item_idx+1, output[0].shape[0], loss.mean().item()))
                    
                    running_loss_value += loss.item()
                    batch_loss += loss
                    
                if batch_idx % batch_checkpoint == batch_checkpoint-1:
                    step += 1
                    wandb.log({"train_loss": running_loss_value / batch_checkpoint , "epoch": epoch + ((batch_idx + 1)/len(train_loader))}, step=step)
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss_value / batch_checkpoint))
                    running_loss_value = 0.0
                
                batch_loss.backward()
                optimizer.step()
            # print("Updated Model weight norms:")
            # for name, param in model.named_parameters():
            #     print(name, param.data.norm())
        # Evaluate on test set
        model.eval()
        
        


        with torch.no_grad():
            for batch_idx, (data, draw_image, output_scale, filenames, ground_truth_keypoints, ground_truth_heatmaps, ground_truth_offsets) in enumerate(test_loader):
                data.cuda()
                data_squeezed = data.squeeze()
                # data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
                output = model(data_squeezed)
                
                # print("**output[0] device: ", output[0].device)
                # print("**ground truth offsets device: ", ground_truth_offsets.device)
                
                # print("**output[0] shape: ", output[0].shape)
                
                
                #iterate through the batch size
                for item_idx, item in enumerate(output[0]):
                                        
                    offsets = output[1][item_idx]
                    displacements_fwd = output[2][item_idx]
                    displacements_bwd = output[3][item_idx]
                    
                    test_heatmaps = item
                    height = test_heatmaps.shape[1]
                    width = test_heatmaps.shape[2]
                    
                    # print("inside item_idx loop offsets shape: ", offsets.shape)
                    # print("item (heatmap) type: ", type(item))
                    
                    pose_scores, keypoint_scores, keypoint_coords, decoded_offsets = decode_pose_from_batch_item(epoch, test_image_path, filenames[item_idx], item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train)
                    
                    appended_text = "test_"
                    

                    draw_coordinates_to_image_file(appended_text, test_image_path, output_dir, output_stride, scale_factor, pose_scores,keypoint_scores, keypoint_coords, filenames[item_idx], include_displacements=False)

                    
                    decoded_offsets = torch.from_numpy(decoded_offsets)
                    decoded_offsets = decoded_offsets.to('cuda')
                    # print("decoded offsets device: ", decoded_offsets.device)
                    # print("ground truth offsets device: ", ground_truth_offsets[item_idx].device)
                    
                    keypoint_coords = torch.from_numpy(keypoint_coords)
                    keypoint_coords = keypoint_coords.to('cuda')
                    
                    # print("keypoint_coords device: ", keypoint_coords.device)
                    # print("ground_truth_keypoints[item_idx] device: ", ground_truth_keypoints[item_idx].device)
                    loss = criterion(test_heatmaps, ground_truth_heatmaps[item_idx], ground_truth_keypoints[item_idx], offsets, ground_truth_offsets[item_idx]).item()
                    
                    
                    test_loss += loss
                    print("inside batch loss value: ", loss)
                    
            test_loss /= len(test_loader.dataset)
            test_loss_value = test_loss.item()
            print("test_loss_value: ", test_loss_value)
            print("step: ", step)
            
            wandb.log({"test_loss": float(test_loss_value)}, step=step)
            
            
            
        # Log epoch duration
        print('Epoch: {} \tTrain Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch+1, running_loss_value, test_loss_value))
                          
        epoch_duration = time.time() - epoch_start_time
        wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)
        epoch_durations.append(epoch_duration)
        
    # Log average epoch duration
    avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
    wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})
    
    print('Training Finished')

def decode_pose_from_batch_item(epoch, image_path, filename, item, offsets, scale_factor, height, width, score_threshold, LOCAL_MAXIMUM_RADIUS, output_stride, displacements_fwd, displacements_bwd, is_train):
    heatmaps = item
    

    # offsets_reshaped = offsets.detach().cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    
    if is_train:
        heatmaps = heatmaps.detach()
        offsets = offsets.detach()
        displacements_fwd = displacements_fwd.detach()
        displacements_bwd = displacements_bwd.detach()
        
    else:
        heatmaps = torch.tensor(heatmaps, requires_grad=is_train)
        
    print("---- in decode pose from batch item --- ") 
    pose_scores, keypoint_scores, keypoint_coords, decoded_offsets = posenet.decode_multi.decode_multiple_poses(
                heatmaps,
                offsets,
                displacements_fwd,
                displacements_bwd,
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=score_threshold)

    
    # print("decoded offsets: ", decoded_offsets)
    
    # Find the indices of poses with scores above the threshold
    valid_indices = np.where(pose_scores >= score_threshold)[0]

    # Filter the pose_scores, keypoint_scores, and keypoint_coords using valid_indices
    pose_scores = pose_scores[valid_indices]
    keypoint_scores = keypoint_scores[valid_indices]
    keypoint_coords = keypoint_coords[valid_indices]
    decoded_offsets = decoded_offsets[valid_indices]
    
    # instance_keypoint_scores, instance_keypoint_coords, displacement_vectors = posenet.decode.decode_pose(root_score, root_id, root_image_coord, heatmaps, offsets_reshaped, output_stride, displacements_fwd_reshaped, displacements_bwd_reshaped)
    
    appended_text = "after_decode_"
    output_dir = "output_after_decode"

    draw_coordinates_to_image_file(appended_text, image_path, output_dir, output_stride, scale_factor, pose_scores, keypoint_scores, keypoint_coords, filename, displacements_fwd, displacements_bwd, include_displacements=True)
                    
    return pose_scores, keypoint_scores, keypoint_coords, decoded_offsets
            

    
def main():
    # Set up training parameters
    batch_size = 2
    learning_rate = 0.001
    num_epochs = 10

    config={
        "epochs": num_epochs,
         "batch_size": batch_size,
         "lr": learning_rate,
         }
    
    with wandb.init(project="posenet", config=config, name='PoseNet 101'):

        #instatiate model 
        model = posenet.load_model(args.model)
        model = model.cuda()
    
        for param in model.parameters():
            param.requires_grad = True
    
        output_stride = model.output_stride     

            
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
    
        is_train = True
    
        train_dataset = PosenetDatasetImage(train_image_path, ground_truth_keypoints_dir, scale_factor=1.0, output_stride=output_stride, train=True)
        test_dataset = PosenetDatasetImage(test_image_path, ground_truth_keypoints_dir, scale_factor=1.0, output_stride=output_stride, train=True)
        
        # when you have updated your dataset, print the mean and std and 
        # replace the Dataset normalization transforms  in class PosenetDatasetImage(Dataset) 
        # mean, std = get_dataset_mean_std(train_dataset)

        

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        
        
        train(model, train_loader, test_loader, criterion, optimizer, num_epochs, output_stride, train_image_path, train_image_path, output_dir, scale_factor, is_train)

        print('Setting up...')
        
        wandb.finish()


if __name__ == "__main__":
    main()