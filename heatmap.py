#create ground truth data by converting points to heatmap
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def convert_ground_truth_kp_to_heatmap(gt_image_keypoints, num_keypoints=17):
    heatmap = np.zeros((33, 33))
    for i in range(0, num_keypoints):
        heatmap += points_to_heatmap(gt_image_keypoints[i][0], gt_image_keypoints[i][1])
    return heatmap


def prepare_ground_truth_data(images_dir, keypoints_dir, num_keypoints=17, heatmaps_dir="heatmaps"):
    # create the output directory if it does not exist
    if not os.path.exists(heatmaps_dir):
        os.mkdir(heatmaps_dir)
    
    # get the list of image files in the directory
    image_files = sorted(os.listdir(images_dir))
    
    keypoint_files = []
    
    for f in image_files:
        image_path = os.path.join(image_dir, f)
        keypoint_path = os.path.join(keypoints_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(keypoint_path):
            keypoint_files.append(keypoint_path)
        else:
            print("Keypoint file does not exist for image:", image_path)
            
    # iterate over the image files
    heatmaps = []
    for i in range(len(image_files)):
        # load the keypoint file
        with open(keypoint_files[i], "r") as f:
            keypoint_data = np.loadtxt(f, delimiter=",")
        
        # convert the keypoints to heatmap
        heatmap = convert_ground_truth_kp_to_heatmap(keypoint_data[:num_keypoints])
        heatmaps.append(heatmap)
    # save the heatmaps to file
    np.save('heatmaps.npy', heatmaps)
    

def main():
    heatmap = points_to_heatmap(4.1, 4.7, kernel_size=11)
    prepare_ground_truth_data()
    
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig('heatmap.png')
    

if __name__ == "__main__":
    main()
