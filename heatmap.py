#create ground truth data by converting points to heatmap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def points_to_heatmap(keypoint_x, keypoint_y, kernel_size=11):      
        
    #create empty heatmap 
    heatmap_size = (33, 33)
    heatmap = np.zeros(heatmap_size)
    
    # Compute a Gaussian kernel centered at the keypoint
    kernel_std = kernel_size / 10
    kernel = cv2.getGaussianKernel(kernel_size, kernel_std)
    kernel = np.outer(kernel, kernel.transpose())
    
    # Add the kernel to the heatmap at the keypoint's coordinates
    xmin = max(0, int(keypoint_x - kernel_size/2))
    xmax = min(heatmap_size[1], int(keypoint_x + kernel_size/2))
    ymin = max(0, int(keypoint_y - kernel_size/2))
    ymax = min(heatmap_size[0], int(keypoint_y + kernel_size/2))
    heatmap[int(ymin):int(ymax), int(xmin):int(xmax)] += kernel[int(ymin-keypoint_y+kernel_size//2):int(ymax-keypoint_y+kernel_size//2), int(xmin-keypoint_x+kernel_size//2):int(xmax-keypoint_x+kernel_size//2)]

    # Normalize the heatmap values
    heatmap /= np.max(heatmap)
    
    return heatmap

def points_to_heatmap(keypoint_x, keypoint_y, kernel_size=11, heatmap_size=(33,33)):    
        
    #create empty heatmap 
    # heatmap_size = (33, 33)
    heatmap = np.zeros(heatmap_size)
    
    # Compute a Gaussian kernel centered at the keypoint
    kernel_std = kernel_size / 10
    kernel = cv2.getGaussianKernel(kernel_size, kernel_std)
    kernel = np.outer(kernel, kernel.transpose())
    
    # Add the kernel to the heatmap at the keypoint's coordinates
    xmin = max(0, int(keypoint_x - kernel_size/2))
    xmax = min(heatmap_size[1], int(keypoint_x + kernel_size/2))
    ymin = max(0, int(keypoint_y - kernel_size/2))
    ymax = min(heatmap_size[0], int(keypoint_y + kernel_size/2))
    heatmap[int(ymin):int(ymax), int(xmin):int(xmax)] += kernel[int(ymin-keypoint_y+kernel_size//2):int(ymax-keypoint_y+kernel_size//2), int(xmin-keypoint_x+kernel_size//2):int(xmax-keypoint_x+kernel_size//2)]

    # Normalize the heatmap values
    heatmap /= np.max(heatmap)
    return heatmap

def convert_ground_truth_kp_to_heatmap(gt_image_keypoints, num_keypoints=17):
    heatmap = np.zeros((33, 33))
    for i in range(0, num_keypoints):
        heatmap += points_to_heatmap(gt_image_keypoints[i][0], gt_image_keypoints[i][1])
    return heatmap


def prepare_ground_truth_data(images_dir, keypoints_dir, num_keypoints=17, heatmaps_dir="heatmaps", heatmap_shape=[33,33]):
    # create the output directory if it does not exist
    if not os.path.exists(heatmaps_dir):
        os.mkdir(heatmaps_dir)
    
    # get the list of image files in the directory
    image_files = sorted(os.listdir(images_dir))
    
    keypoint_files = []
    
     # iterate over the image files
    for image_file in image_files:
                
        # construct the paths to the image and keypoint files
        image_path = os.path.join(images_dir, image_file)
        keypoint_path = os.path.join(keypoints_dir, os.path.splitext(image_file)[0] + ".txt")

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
                center_x = float(parts[1]) * heatmap_shape[1]
                center_y = float(parts[2]) * heatmap_shape[0]
                # width = float(parts[3]) * heatmap_shape[1]
                # height = float(parts[4]) * heatmap_shape[0]
                
                #Ignore the last keypoint which is the bounding box of the person
                if keypoint_id != num_keypoints: 
                    keypoints[keypoint_id] = np.array([center_x, center_y])
                
            heatmaps = np.zeros((num_keypoints, heatmap_shape[0], heatmap_shape[1]))
            
            for i, keypoint_coord in enumerate(keypoints):
                heatmap = points_to_heatmap(keypoint_coord[i][0], keypoint_coord[i][1], kernel_size=11, heatmap_size=(33,33))
                heatmaps.append(heatmap)
        

        # create a folder for the heatmaps of this image
        output_dir = os.path.join(heatmaps_dir, os.path.splitext(image_file)[0])
        os.makedirs(output_dir, exist_ok=True)

        # save the heatmaps to separate files in the output folder
        for i in range(num_keypoints):
            output_file = os.path.join(output_dir, f"heatmap_{i}.npy")
            output_image = os.path.join(output_dir, f"heatmap_{i}.png")
            np.save(output_file, heatmaps[i])
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.savefig(output_image)
    
    

def main():
    heatmap = points_to_heatmap(4.1, 4.7, kernel_size=11)
    prepare_ground_truth_data('images_train', 'labels_train', num_keypoints=17, heatmaps_dir="heatmaps", heatmap_shape=[33,33])
        
    
    
    

if __name__ == "__main__":
    main()
