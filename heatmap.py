#create ground truth data by converting points to heatmap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal


def points_to_heatmap(keypoint_x, keypoint_y, kernel_size=11, heatmap_size=(33,33)):    
        
    #create empty heatmap 
    # heatmap_size = (33, 33)
    heatmap = np.zeros(heatmap_size)
    print("heatmap size: ", heatmap.size)
    
    # Compute a Gaussian kernel centered at the keypoint
    kernel_std = kernel_size / 10
    kernel = cv2.getGaussianKernel(kernel_size, kernel_std)
    kernel = np.outer(kernel, kernel.transpose())
    
    # kernel = multivariate_normal(mean=[0, 0], cov=np.eye(2)*(kernel_std)**2)

    
    # Add the kernel to the heatmap at the keypoint's coordinates
    # xmin = max(0, int(keypoint_x - kernel_size/2))
    # xmax = min(heatmap_size[1], int(keypoint_x + kernel_size/2))
    # ymin = max(0, int(keypoint_y - kernel_size/2))
    # ymax = min(heatmap_size[0], int(keypoint_y + kernel_size/2))
    
    xmin = max(int(keypoint_x - kernel_size//2), 0)
    xmax = min(int(keypoint_x + kernel_size//2 + 1), heatmap_size[1])
    ymin = max(int(keypoint_y - kernel_size//2), 0)
    ymax = min(int(keypoint_y + kernel_size//2 + 1), heatmap_size[0])
    
    #how much it is deviating from keypoint_x 
    #if at keypoint_x, then kernel will be at kernel_size//2
    
    kernel_xmin = max(0, kernel_size//2 - int(keypoint_x) - xmin) 
    kernel_xmax = min(kernel_size, kernel_size//2 + xmax - int(keypoint_x))
    kernel_ymin = max(0, kernel_size//2 - int(keypoint_y) + ymin)
    kernel_ymax = min(kernel_size, kernel_size//2 + ymax - int(keypoint_y))
                      
#     kernel_xmin = max(int(kernel_size//2 - keypoint_x), 0)
#     kernel_xmax = min(int(kernel_size//2 + heatmap_size[1] - keypoint_x), kernel_size)
#     kernel_ymin = max(int(kernel_size//2 - keypoint_y), 0)
#     kernel_ymax = min(int(kernel_size//2 + heatmap_size[0] - keypoint_y), kernel_size)
    
    
    print("xmin: ", xmin)
    print("xmax: ", xmax)
    print("ymin: ", ymin)
    print("ymax: ", ymax)
    
    print("kernel_xmin: ", kernel_xmin)
    print("kernel_xmax: ", kernel_xmax)
    print("kernel_ymin: ", kernel_ymin)
    print("kernel_ymax: ", kernel_ymax)
    
    # heatmap[int(ymin):int(ymax), int(xmin):int(xmax)] += kernel[int(ymin-keypoint_y+kernel_size//2):int(ymax-keypoint_y+kernel_size//2), int(xmin-keypoint_x+kernel_size//2):int(xmax-keypoint_x+kernel_size//2)]

    heatmap[ymin:ymax, xmin:xmax] += kernel[kernel_ymin:kernel_ymax, kernel_xmin:kernel_xmax]

    # Normalize the heatmap values
    heatmap /= np.max(heatmap)
    return heatmap

def prepare_ground_truth_data(images_dir, keypoints_dir, num_keypoints=17, heatmaps_dir="heatmaps", heatmap_shape=[33,33]):
    # create the output directory if it does not exist
    if not os.path.exists(heatmaps_dir):
        os.mkdir(heatmaps_dir)
    
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
        
        # load the keypoint file
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
                if keypoint_id != num_keypoints:
                    new_keypoint_id = index_map[keypoint_id]
                    keypoints[keypoint_id] = np.array([center_x, center_y])
                
            heatmaps = np.zeros((num_keypoints, heatmap_shape[0], heatmap_shape[1]))
            
            for i, keypoint_coord in enumerate(keypoints):
                print("i: ", i)
                print("Keypoint_coord x: ", keypoint_coord[0])
                print("Keypoint_coord y: ", keypoint_coord[1])
                heatmap = points_to_heatmap(keypoint_coord[0], keypoint_coord[1], kernel_size=11, heatmap_size=(33,33))
                
                heatmaps[i] = heatmap
        

        # create a folder for the heatmaps of this image
        output_dir = os.path.join(heatmaps_dir, os.path.splitext(image_file)[0])
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'npy'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)
        

        # save the heatmaps to separate files in the output folder
        for i in range(num_keypoints):
            output_file = os.path.join(output_dir, 'npy', f"heatmap_{i}.npy")
            output_image = os.path.join(output_dir, 'png', f"heatmap_{i}.png")
            np.save(output_file, heatmaps[i])
            plt.imshow(heatmaps[i], cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(output_image)
            plt.clf()
            
def remap_keypoint_coordinates_index(original_names, new_order_names):
    
    # create a dictionary that maps original indices to new indices
    index_map = {}
    for i, name in enumerate(original_names):
        index_map[i] = new_order_names.index(name)
    return index_map

    
    

    
def main():
    # heatmap = points_to_heatmap(4.1, 4.7, kernel_size=11)
    prepare_ground_truth_data('images_train', 'labels_train', num_keypoints=17, heatmaps_dir="heatmaps_train", heatmap_shape=[33,33])
    

if __name__ == "__main__":
    main()
