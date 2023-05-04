#create ground truth data by converting points to heatmap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from skimage.morphology import dilation, disk
from posenet import constants as constants
import torch
import json
import re

def prepare_ground_truth_data(images_dir, keypoints_dir, num_keypoints=17, heatmaps_dir="heatmaps", heatmap_shape=[33,33], keypoints_updated_dir="keypoints_updated"):
    # create the output directory if it does not exist
    if not os.path.exists(heatmaps_dir):
        os.makedirs(heatmaps_dir)
        
    if not os.path.exists(keypoints_updated_dir):
        os.makedirs(keypoints_updated_dir)
    
    # get the list of image files in the directory
    image_files = sorted(os.listdir(images_dir))
    
    keypoint_files = []
    
    #remapping roboflow keypoint coordinates index to posenet keypoint coordinates index
#     original_names = ['0-nose', '1-leftEye', '10-rightWrist', '11-leftHip', '12-rightHip', '13-leftKnee', '14-rightKnee', '15-leftAnkle', '16-rightAnkle', '17-person', '2-rightEye', '3-leftEar', '4-rightEar', '5-leftShoulder', '6-rightShoulder', '7-leftElbow', '8-rightElbow', '9-leftWrist']
#     new_order_names = ['0-nose', '1-leftEye', '2-rightEye', '3-leftEar', '4-rightEar', '5-leftShoulder', '6-rightShoulder', '7-leftElbow', '8-rightElbow', '9-leftWrist', '10-rightWrist', '11-leftHip', '12-rightHip', '13-leftKnee', '14-rightKnee', '15-leftAnkle', '16-rightAnkle', '17-person']
        
    #prepare index map to reindex the keypoints
    # index_map = remap_keypoint_coordinates_index(original_names, new_order_names)
        
     # iterate over the image files
    for image_file in image_files:
                
        # construct the paths to the image and keypoint files
        image_path = os.path.join(images_dir, image_file)
        keypoint_path = os.path.join(keypoints_dir, os.path.splitext(image_file)[0] + ".json")
        
        print("=== IMAGE FILE ===")
        print(image_file)
        # check if the keypoint file exists
        if not os.path.exists(keypoint_path):
            print("Keypoint file does not exist for image:", image_path)
            continue

        # load the image
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        image_scale_x = heatmap_shape[1] / image_width
        image_scale_y = heatmap_shape[0] / image_height
        
        # create a new directory for the image
        image_dir = os.path.join(keypoints_updated_dir, os.path.splitext(image_file)[0])
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        # load the original keypoints
        # keypoints scaled to heatmap shape [33, 33]
        # keypoints shape [num_poses, 17, 2]
        keypoints = keypoint_path_to_heatmap_keypoints(keypoint_path, num_keypoints, heatmap_shape)

        
#         original_keypoints_file = os.path.join(image_dir, os.path.splitext(image_file)[0] + "_keypoints.txt")
#         np.savetxt(original_keypoints_file, keypoints, delimiter=",")
        
        # create the heatmaps from the original keypoints
        # heatmaps of shape [num_poses, 17, 33,33] 
        heatmaps = load_keypoints(keypoints, num_keypoints, heatmap_shape)
        
        num_poses = heatmaps.shape[0]
        
        # generate the keypoints from the heatmaps
        # generated_keypoints are scaled to heatmap shape size [33,33]
        # generated_keypoints shape [num_poses, 17, 2]
        generated_keypoints = generated_keypoints_from_heatmaps(heatmaps) 
        
        # combine keypoints and generated keypoints from all poses into a single array
        all_keypoints = np.concatenate(keypoints, axis=0)
        all_generated_keypoints = np.concatenate(list(generated_keypoints), axis=0)

        #save the keypoints
        #keypoints of all poses are saved in the same .txt file in chronological order
        #the number of rows / 17 is the number of poses
        keypoints_file = os.path.join(image_dir, os.path.splitext(image_file)[0] + "_keypoints.txt")
        np.savetxt(keypoints_file, all_keypoints, delimiter=",")
            
        #save generated keypoints
        #keypoints of all poses are saved in the same .txt file in chronological order
        #the number of rows / 17 is the number of poses 
        generated_keypoints_file = os.path.join(image_dir, os.path.splitext(image_file)[0] + "_generated.txt")
        np.savetxt(generated_keypoints_file, all_generated_keypoints, delimiter=",")

        
        keypoints = torch.from_numpy(keypoints)
        
        # create the ground truch offset vectors
        offset_vectors = generate_offset_vectors(keypoints, generated_keypoints)
        
        for pose_idx in range(num_poses):
            save_offset_vectors(offset_vectors, image_file, pose_idx, num_keypoints, heatmaps_dir)
            save_heatmaps(heatmaps, image_file, pose_idx, num_keypoints, heatmaps_dir)
            
        # save_offset_vectors(offset_vectors, image_file, num_keypoints, heatmaps_dir)
        
        
def generate_offset_vectors(keypoints, generated_keypoints):
    offset_vectors = keypoints - generated_keypoints
    return offset_vectors
        
def points_to_heatmap(keypoint_x, keypoint_y, kernel_size=11, heatmap_size=(33,33)):
    if keypoint_x == 0 and keypoint_y == 0:
        return np.zeros(heatmap_size)

    # Create empty heatmap
    heatmap = np.zeros(heatmap_size)

    # Compute a Gaussian kernel centered at the keypoint
    kernel_std = kernel_size / 10
    kernel = cv2.getGaussianKernel(kernel_size, kernel_std)
    kernel = np.outer(kernel, kernel.transpose())

    xmin = max(int(keypoint_x - kernel_size//2), 0)
    xmax = min(int(keypoint_x + kernel_size//2 + 1), heatmap_size[1])
    ymin = max(int(keypoint_y - kernel_size//2), 0)
    ymax = min(int(keypoint_y + kernel_size//2 + 1), heatmap_size[0])

    kernel_xmin = max(0, kernel_size//2 - int(keypoint_x) - xmin)
    kernel_xmax = min(kernel_size, kernel_size//2 + xmax - int(keypoint_x))
    kernel_ymin = max(0, kernel_size//2 - int(keypoint_y) + ymin)
    kernel_ymax = min(kernel_size, kernel_size//2 + ymax - int(keypoint_y))

    heatmap[ymin:ymax, xmin:xmax] += kernel[kernel_ymin:kernel_ymax, kernel_xmin:kernel_xmax]

    # Normalize the heatmap values
    heatmap /= np.max(heatmap)
    return heatmap


#create generated keypoints from heatmaps
# generated keypoints are created by running a sigmoid function on the heatmap and then argmax to find the x and y coordinates 
def generated_keypoints_from_heatmaps(heatmaps):
    num_poses, num_keypoints, height, width = heatmaps.shape

    # find the index of the maximum value along the last two dimensions
    heatmaps = torch.from_numpy(heatmaps)
    heatmaps = torch.sigmoid(heatmaps)

    max_idxs = heatmaps.view(num_poses, num_keypoints, -1).argmax(dim=-1)
    max_y = torch.div(max_idxs, height, rounding_mode='floor')
    max_x = max_idxs % width

    generated_keypoints = torch.cat([max_x.unsqueeze(-1), max_y.unsqueeze(-1)], dim=-1)
    # print("--generated keypoints --")
    # print("generated keypoints shape: ", generated_keypoints.shape)
    # print(generated_keypoints)
    
    return generated_keypoints

    
def save_offset_vectors(offset_vectors, image_file, pose_idx, num_keypoints, heatmaps_dir):
    output_dir = os.path.join(heatmaps_dir, os.path.splitext(image_file)[0])
    offset_vectors_file = os.path.join(output_dir, os.path.splitext(image_file)[0] + f"_offset_vectors_pose_{pose_idx}.txt")
    np.savetxt(offset_vectors_file, offset_vectors[pose_idx], fmt="%f", delimiter=",")

        
import matplotlib.pyplot as plt

def save_heatmaps(heatmaps, image_file, pose_idx, num_keypoints, heatmaps_dir="heatmaps"):
    # create a folder for the heatmaps of this image
    output_dir = os.path.join(heatmaps_dir, os.path.splitext(image_file)[0], f"pose_{pose_idx}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'npy'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)
    
    for i in range(num_keypoints):
        output_file = os.path.join(output_dir, 'npy', f"heatmap_{i}.npy")
        output_image = os.path.join(output_dir, 'png', f"heatmap_{i}.png")
        
        np.save(output_file, heatmaps[pose_idx][i])
        plt.imshow(heatmaps[pose_idx][i], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.savefig(output_image)
        plt.clf()
        

# load the keypoints from the image file
# input: keypoints shape [n, 17, 2]
def load_keypoints(keypoints, num_keypoints, heatmap_shape):
    all_heatmaps = []
    
    for person_keypoints in keypoints:
        heatmaps = np.zeros((num_keypoints, heatmap_shape[0], heatmap_shape[1]))

        for i, keypoint_coord in enumerate(person_keypoints):
            heatmap = points_to_heatmap(keypoint_coord[0], keypoint_coord[1], kernel_size=11, heatmap_size=(33,33))
            heatmaps[i] = heatmap

        all_heatmaps.append(heatmaps)

    all_heatmaps = np.stack(all_heatmaps, axis=0)
    
    return all_heatmaps

# load the keypoints from the keypoint file
# keypoints are scaled to heatmap size
def keypoint_path_to_heatmap_keypoints(keypoint_path, num_keypoints, heatmap_shape):
    #TODO : map the right keypoints to the right order?? 
    
    with open(keypoint_path, "r") as f:
        data = json.load(f)
        
        annotations = data["annotations"]
        
        image_height = data["metadata"]["system"]["height"]
        image_width = data["metadata"]["system"]["width"]
        
        # Calculate the scaling factors
        x_scale = heatmap_shape[1] / image_width
        y_scale = heatmap_shape[0] / image_height
        
        # Create a dictionary to store poses and their keypoints
        poses = []
        keypoints_list = []
        
        # Create a mapping from keypoint labels to their indices in constants.PART_NAMES
        label_to_index
        
        label_to_index = {add_space_before_capital(name).lower(): i for i, name in enumerate(constants.PART_NAMES)}
        print("label_to_index: ", label_to_index)
        
        # Iterate over the annotations
        pose_count = 0
        for annotation in annotations:
            if annotation["type"] == "pose":
                pose_id = annotation["id"]
                pose_entry = {"id": pose_id, "keypoints": [(-1, -1)] * len(constants.PART_NAMES)}
                poses.append(pose_entry)
                
            elif annotation['type'] == 'point':
                parent_id = annotation['metadata']['system']['parentId']
                keypoint_label = annotation['label']
                keypoint_id = label_to_index[keypoint_label.lower()]
                x = annotation['coordinates']['x'] * x_scale
                y = annotation['coordinates']['y'] * y_scale
                keypoints_list.append((parent_id, keypoint_id, x, y))
                
        for parent_id, keypoint_id, x, y in keypoints_list:
            for pose in poses:
                if pose['id'] == parent_id:
                    pose['keypoints'][keypoint_id] = (x, y)
                    break
         # Filter out poses with all keypoints having value (-1, -1)
        valid_poses = [pose for pose in poses if not all(kp == (-1, -1) for kp in pose['keypoints'])]
        keypoints_arrays = [np.array(pose['keypoints']) for pose in valid_poses]

        
        # Convert keypoints_arrays to the desired format [n, 17, x, y]
        keypoints_final = np.empty((len(keypoints_arrays), num_keypoints, 2), dtype=float)
        for i, pose_keypoints in enumerate(keypoints_arrays):
            for j, keypoint in enumerate(pose_keypoints):
                keypoints_final[i, j] = keypoint
            
        print("file name: ", f)
        # print("keypoints_final: ")
        # print(keypoints_final) 
        print("keypoints_final shape: ", keypoints_final.shape)
            
        return keypoints_final

#         keypoints = np.zeros((num_keypoints,2))
            
#         for line in f:
#             parts = line.strip().split()
#             keypoint_id = int(parts[0])
#             print("normalized x: ", parts[1])
#             print("normalized y: ", parts[2])
                
#             center_x = float(parts[1]) * heatmap_shape[1]
#             center_y = float(parts[2]) * heatmap_shape[0]
                
#             print("center_x: ", center_x)
#             print("center_y: ", center_y)
#             # width = float(parts[3]) * heatmap_shape[1]
#             # height = float(parts[4]) * heatmap_shape[0]
                
#             #Ignore the last keypoint which is the bounding box of the person
#             new_keypoint_id = index_map[keypoint_id]
#             if new_keypoint_id != num_keypoints:
#                 keypoints[new_keypoint_id] = np.array([center_x, center_y])
            

def add_space_before_capital(s):
    return re.sub(r'([A-Z])', r' \1', s)


# def remap_keypoint_coordinates_index(original_names, new_order_names):
    
#     # create a dictionary that maps original indices to new indices
#     index_map = {}
#     for i, name in enumerate(original_names):
#         index_map[i] = new_order_names.index(name)
#     return index_map


def load_ground_truth_data(image_file_names, keypoints_updated_dir):
    keypoints_list = []
    heatmaps_list = []
    offset_vectors_list = []

    for image_file_name in image_file_names:
        image_file_dir = os.path.join(keypoints_updated_dir, image_file_name)
        pose_keypoints_list = []
              
        keypoints_file = os.path.join(image_file_dir, image_file_name + "_keypoints.txt")
        print("load ground truth keypoints_file", keypoints_file)
        generated_keypoints_file = os.path.join(image_file_dir, image_file_name + "_generated.txt")
        print("load ground truth generated_keypoints file", generated_keypoints_file)
        
        # Load the flattened keypoints
        keypoints_flat = np.loadtxt(keypoints_file, delimiter=",")
        generated_keypoints_flat = np.loadtxt(generated_keypoints_file, delimiter=",")

        # Determine the number of poses based on the number of rows in the flattened keypoints
        num_poses = int(keypoints_flat.shape[0] / 17)
        
        # Reshape the keypoints and generated keypoints arrays to have the correct dimensions
        keypoints = keypoints_flat.reshape(num_poses, 17, 2)
        generated_keypoints = generated_keypoints_flat.reshape(num_poses, 17, 2)

        # Generate the heatmaps from the keypoints
        heatmaps = load_keypoints(keypoints, num_keypoints=17, heatmap_shape=(33, 33))

        keypoints_list.append(keypoints)
        heatmaps_list.append(heatmaps)
        offset_vectors_list.append(generate_offset_vectors(keypoints, generated_keypoints))
    
    print("--type--")
    print("heatmaps list len: ", len(heatmaps_list))
    print("heatmaps list [0] shape: ", heatmaps_list[0].shape)
    print("offset vectors list shape: ", offset_vectors_list[0].shape)
    
    # keypoints_list = [np.array(keypoints, dtype=np.float32) for keypoints in keypoints_list]
    # heatmaps_list = [np.array(heatmaps, dtype=np.float32) for heatmaps in heatmaps_list]
    # offset_vectors_list = [np.array(offset_vectors, dtype=np.float32) for offset_vectors in offset_vectors_list]
    
    num_images = len(keypoints_list)
    keypoints_padded = np.full((num_images, 15, 17, 2), -1)
    heatmaps_padded = np.full((num_images, 15, 17, 33, 33), -1)
    offset_vectors_padded = np.full((num_images, 15, 17, 2), -1)
    
    for image in range(num_images): 
        num_poses = keypoints_list[image].shape[0]
        keypoints_padded[image, :num_poses, :, :] = keypoints_list[image]
        heatmaps_padded[image, :num_poses, :, :] = heatmaps_list[image]
        offset_vectors_padded[image, :num_poses, :, :] = offset_vectors_list[image]
        
    
    keypoints_padded = torch.from_numpy(keypoints_padded).cuda()
    
    heatmaps_padded = torch.from_numpy(heatmaps_padded).cuda()
    offset_vectors_padded = torch.from_numpy(offset_vectors_padded).cuda()
    
    
    print("keypoints_list shape: ", keypoints_padded.shape)
    print("heatmaps_list shape: ", heatmaps_padded.shape)
    print("offset_vectors shape: ", offset_vectors_padded.shape)

    return keypoints_padded, heatmaps_padded, offset_vectors_padded


def image_file_name(file):
    return os.path.splitext(os.path.basename(file))[0]

    
def main():
    # heatmap = points_to_heatmap(4.1, 4.7, kernel_size=11)
    prepare_ground_truth_data('images_train', 'labels_train', num_keypoints=17, heatmaps_dir="heatmaps_train", heatmap_shape=[33,33], keypoints_updated_dir="keypoints_updated")

if __name__ == "__main__":
    main()
