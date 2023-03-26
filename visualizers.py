#import related libraries
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import posenet
from posenet.constants import *
matplotlib.use('Agg')



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


def draw_coordinates_to_image_file(appended_text, image_path, output_dir, output_stride, scale_factor, pose_scores, keypoint_scores, keypoint_coords, filename, displacements_fwd=None, displacements_bwd=None, include_displacements=False):
    
    print("------- inside draw_coordinates_to_image_file")
    print("appended_text: ", appended_text)
    print("image_path: ", image_path)
    print("output_dir: ", output_dir)
    
    input_image, draw_image, output_scale = posenet.read_imgfile(
        os.path.join(image_path, filename), scale_factor, output_stride=output_stride)

    
    original_image = cv2.imread(os.path.join(image_path, filename))
    original_height, original_width, _ = original_image.shape

    # Get the current dimensions of the draw_image
    current_height, current_width, _ = draw_image.shape
    # print(f"Current dimensions: {current_width}x{current_height}")
    
    # Clone keypoint_coords tensor to avoid in-place operation error
    keypoint_coords = keypoint_coords.copy()
    keypoint_coords = keypoint_coords.astype(np.float32)

    
    # Convert keypoint_coords to a NumPy array
    # keypoint_coords = keypoint_coords.detach().cpu().numpy()
    
    
    # keypoint_coords = keypoint_coords * output_scale * output_stride
    
    # Add extra dimension to keypoint_scores and keypoint_coords
    #if keypoint_scores is single pose with dimension of 1, add extra dimension
    # if (len(keypoint_scores.shape) == 1):
        
        # print("DRAW coords keypoint_scores original shape: ", keypoint_scores.shape) 
        # print("DRAW coords keypoint_coords original shape: ", keypoint_coords.shape) 
    min_pose_score = 1.0
        
    # num_instances = keypoint_scores.shape[0] # Get the number of instances
    # instance_scores = [min_pose_score for _ in range(num_instances)]
    
    keypoint_coords *= output_scale

    draw_image = posenet.draw_skel_and_kp(
        draw_image, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.25, min_part_score=0.25)


    # Repeat the keypoint scores and keypoint coordinates as many times as the number of instances
    # keypoint_scores = np.tile(keypoint_scores[np.newaxis, :], (num_instances, 1))
    # keypoint_coords = np.tile(keypoint_coords[np.newaxis, :, :], (num_instances, 1, 1))
    
    # Create a list of instance scores with the same length as the number of keypoints

    # Resize the draw_image back to the original dimensions
    draw_image = cv2.resize(draw_image, (original_width, original_height))

#     if include_displacements:
#         print("displacements_fwd shape: ", displacements_fwd.shape)
#         # displacements_fwd_np = displacements_fwd.cpu().numpy()
#         # displacement_shape = displacements_fwd_np[0].shape

#         # Create a mesh grid for the quiver plot of the displacements
#         # Y, X = np.mgrid[0:displacement_shape[0], 0:displacement_shape[1]]
        

#         # Overlay the displacements onto the draw_image
#         # draw_image_quiver = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
#         # fig, ax = plt.subplots()
#         # ax.imshow(draw_image_quiver)

#         print("Before draw_displacement_vectors")

#         # fig, ax = draw_displacement_vectors(draw_image, keypoint_coords, displacements_fwd, output_stride, scale_factor)
        
#         # print("After loop for keypoint pairs")  # Add this print statement
#         # ax.axis('off')
#         # print("Before saving the plot")
#         # plt.savefig(os.path.join(output_dir, f"keypoints_{appended_text}_{filename}"), dpi=100)
#         print("After saving the plot")
#         # plt.close(fig)
#         print("After closing the plot")

    # Draw the keypoints and skeleton on the draw_image
    abs_output_dir = os.path.abspath(output_dir)
    print("Absolute output_dir: ", abs_output_dir)
    
    # Update image_path to include epoch number
    image_filename = f"{appended_text}_{filename}"
    # Get the absolute path of the saved image file
    print("image_filename: ", image_filename)
    cv2.imwrite(os.path.join(output_dir, image_filename), draw_image)
    

def draw_displacement_vectors(image, keypoints, displacements, output_stride, scale_factor):
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    print("inside draw_displacement_vectors ====")
    print("displacements shape: ", displacements.shape)
    
    for edge_id, (source_keypoint_id, target_keypoint_id) in enumerate(PARENT_CHILD_TUPLES):
        if source_keypoint_id < keypoints.shape[0] and target_keypoint_id < keypoints.shape[0]:
            source_keypoint = keypoints[source_keypoint_id] * scale_factor
            target_keypoint = keypoints[target_keypoint_id] * scale_factor
            displacement_vector = displacements[edge_id] * scale_factor
            ax.arrow(source_keypoint[1], source_keypoint[0],
                     displacement_vector[1], displacement_vector[0],
                     head_width=1.5, head_length=2, fc='blue', ec='blue', linewidth=1.5)

    return fig, ax

