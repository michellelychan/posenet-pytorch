#create ground truth data by converting points to heatmap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from skimage.morphology import dilation, disk
import torch

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
    original_names = ['0-nose', '1-leftEye', '10-rightWrist', '11-leftHip', '12-rightHip', '13-leftKnee', '14-rightKnee', '15-leftAnkle', '16-rightAnkle', '17-person', '2-rightEye', '3-leftEar', '4-rightEar', '5-leftShoulder', '6-rightShoulder', '7-leftElbow', '8-rightElbow', '9-leftWrist']
    new_order_names = ['0-nose', '1-leftEye', '2-rightEye', '3-leftEar', '4-rightEar', '5-leftShoulder', '6-rightShoulder', '7-leftElbow', '8-rightElbow', '9-leftWrist', '10-rightWrist', '11-leftHip', '12-rightHip', '13-leftKnee', '14-rightKnee', '15-leftAnkle', '16-rightAnkle', '17-person']
        
    #prepare index map to reindex the keypoints
    index_map = remap_keypoint_coordinates_index(original_names, new_order_names)
        
     # iterate over the image files
    for image_file in image_files:
                
        # construct the paths to the image and keypoint files
        image_path = os.path.join(images_dir, image_file)
        keypoint_path = os.path.join(keypoints_dir, os.path.splitext(image_file)[0] + ".txt")
        
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
        keypoints = keypoint_path_to_heatmap_keypoints(keypoint_path, num_keypoints, heatmap_shape, index_map)
        original_keypoints_file = os.path.join(image_dir, os.path.splitext(image_file)[0] + "_keypoints.txt")
        np.savetxt(original_keypoints_file, keypoints, delimiter=",")
        
        # create the heatmaps from the original keypoints
        heatmaps = load_keypoints(keypoints, num_keypoints, heatmap_shape)
        
        # generate the keypoints from the heatmaps
        generated_keypoints = generated_keypoints_from_heatmaps(heatmaps) 
        generated_keypoints_file = os.path.join(image_dir, os.path.splitext(image_file)[0] + "_generated.txt")
        np.savetxt(generated_keypoints_file, generated_keypoints, delimiter=",")
        
        print("HEATMAPS SHAPE")
        print(heatmaps.shape)
        
        
        keypoints = torch.from_numpy(keypoints)
        # create the ground truch offset vectors
        offset_vectors = generate_offset_vectors(keypoints, generated_keypoints)

        save_heatmaps(heatmaps, image_file, num_keypoints, heatmaps_dir)
        save_offset_vectors(offset_vectors, image_file, num_keypoints, heatmaps_dir)
        
        
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
    height = heatmaps.shape[1]
    width = heatmaps.shape[2]
                
    # find the index of the maximum value along the last two dimensions
    heatmaps = torch.from_numpy(heatmaps)
    heatmaps = torch.sigmoid(heatmaps)
        
    max_idxs = heatmaps.view(17, 1, -1).argmax(dim=-1)
    max_y = torch.div(max_idxs, width, rounding_mode='floor')
    max_x = max_idxs % width
        
    generated_keypoints = torch.cat([max_x, max_y], dim=1)

    return generated_keypoints

def save_offset_vectors(offset_vectors, image_file, num_keypoints, heatmaps_dir):
    output_dir = os.path.join(heatmaps_dir, os.path.splitext(image_file)[0])
    offset_vectors_file = os.path.join(output_dir, os.path.splitext(image_file)[0] + "_offset_vectors.txt")
    #print("=== OFFSET VECTORS FILE ===")
    #print(offset_vectors_file)
    pose_offset_vectors = np.expand_dims(offset_vectors, axis=0)
    np.savetxt(offset_vectors_file, offset_vectors, fmt="%f", delimiter=",")
        
def save_heatmaps(heatmaps, image_file, num_keypoints, heatmaps_dir="heatmaps"):
    # create a folder for the heatmaps of this image
    output_dir = os.path.join(heatmaps_dir, os.path.splitext(image_file)[0])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'npy'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)
    
    # save the heatmaps in a single txt file
    heatmaps_file = os.path.join(output_dir, os.path.splitext(image_file)[0] + "_heatmap.txt")
    np.savetxt(heatmaps_file, heatmaps.reshape(num_keypoints, -1), fmt="%f", delimiter=",")
    

    # save the heatmaps to separate files in the output folder
    for i in range(num_keypoints):
        output_file = os.path.join(output_dir, 'npy', f"heatmap_{i}.npy")
        output_image = os.path.join(output_dir, 'png', f"heatmap_{i}.png")
        np.save(output_file, heatmaps[i])
        plt.imshow(heatmaps[i], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.savefig(output_image)
        plt.clf()
              

# load the keypoints from the image file
def load_keypoints(keypoints, num_keypoints, heatmap_shape):        
    heatmaps = np.zeros((num_keypoints, heatmap_shape[0], heatmap_shape[1]))

            
    for i, keypoint_coord in enumerate(keypoints):
        print("i: ", i)
        print("Keypoint_coord x: ", keypoint_coord[0])
        print("Keypoint_coord y: ", keypoint_coord[1])
        heatmap = points_to_heatmap(keypoint_coord[0], keypoint_coord[1], kernel_size=11, heatmap_size=(33,33))

        heatmaps[i] = heatmap

    return heatmaps

# load the keypoints from the keypoint file
# keypoints are scaled to heatmap size
def keypoint_path_to_heatmap_keypoints(keypoint_path, num_keypoints, heatmap_shape, index_map):
    with open(keypoint_path, "r") as f:
        keypoints = np.zeros((num_keypoints,2))
            
        for line in f:
            parts = line.strip().split()
            keypoint_id = int(parts[0])
            print("normalized x: ", parts[1])
            print("normalized y: ", parts[2])
                
            center_x = float(parts[1]) * heatmap_shape[1]
            center_y = float(parts[2]) * heatmap_shape[0]
                
            print("center_x: ", center_x)
            print("center_y: ", center_y)
            # width = float(parts[3]) * heatmap_shape[1]
            # height = float(parts[4]) * heatmap_shape[0]
                
            #Ignore the last keypoint which is the bounding box of the person
            new_keypoint_id = index_map[keypoint_id]
            if new_keypoint_id != num_keypoints:
                keypoints[new_keypoint_id] = np.array([center_x, center_y])
            
    return keypoints



def remap_keypoint_coordinates_index(original_names, new_order_names):
    
    # create a dictionary that maps original indices to new indices
    index_map = {}
    for i, name in enumerate(original_names):
        index_map[i] = new_order_names.index(name)
    return index_map


def load_ground_truth_data(image_file_names, keypoints_updated_dir):
    keypoints_list = []
    heatmaps_list = []
    offset_vectors_list = []

    for image_file_name in image_file_names:
        image_file_dir = os.path.join(keypoints_updated_dir, image_file_name)
        keypoints_file = os.path.join(image_file_dir, image_file_name + "_keypoints.txt")
        print("load ground truth keypoints_file", keypoints_file)
        generated_keypoints_file = os.path.join(image_file_dir, image_file_name + "_generated.txt")
        print("load ground truth generated_keypoints file", generated_keypoints_file)
        keypoints = np.loadtxt(keypoints_file, delimiter=",")
        generated_keypoints = np.loadtxt(generated_keypoints_file, delimiter=",")

        # Generate the heatmaps from the keypoints
        heatmaps = load_keypoints(keypoints, num_keypoints=17, heatmap_shape=(33, 33))

        keypoints_list.append(keypoints)
        heatmaps_list.append(heatmaps)
        offset_vectors_list.append(generate_offset_vectors(keypoints, generated_keypoints))
        print("keypoints_list length: ", len(keypoints_list))
        print("heatmaps_list length: ", len(heatmaps_list))
        print("offset_vectors length: ", len(offset_vectors_list))

    return keypoints_list, heatmaps_list, offset_vectors_list

def image_file_name(file):
    return os.path.splitext(os.path.basename(file))[0]

    
def main():
    # heatmap = points_to_heatmap(4.1, 4.7, kernel_size=11)
    prepare_ground_truth_data('images_train', 'labels_train', num_keypoints=17, heatmaps_dir="heatmaps_train", heatmap_shape=[33,33], keypoints_updated_dir="keypoints_updated")

if __name__ == "__main__":
    main()
