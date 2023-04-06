import numpy as np
import torch
import torch.nn.functional as F


from posenet.constants import *


def traverse_to_targ_keypoint(
        edge_id, source_keypoint, target_keypoint_id, scores, offsets, output_stride, displacements
):
    height = scores.shape[1]
    width = scores.shape[2]

    source_keypoint_indices = np.clip(
        np.round(source_keypoint / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)
    
    #make source_keypoint addable to displacements
    #print source_keypoint type is it np array or tensor
    # print("source_keypoint type: ", type(source_keypoint))
    # print("displacements type: ", type(displacements))
    # print("displacements shape: ", displacements.shape)
    # print("source_keypoint shape: ", source_keypoint.shape)
    
#     print("displacements_value shape: ", displacement_value.shape)
#     print("displacements_value value: ", displacement_value) 

#     print("source_keypoing_indices[0]: ", source_keypoint_indices[0])
#     print("source_keypoint_indices[1]: ", source_keypoint_indices[1])
#     print("edge id: ", edge_id)
    
    # displaced_point = source_keypoint + displacement_value
    
    # print("inside traverse_to_targ_keypoint ******")
    # print("source_keypoint shape: ", source_keypoint.shape)
    # print("source_keypoint: ", source_keypoint)
    # print("displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]] shape: ", displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]].shape)
    # print("displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]]: ", displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]])
    displacement_vector = displacements[edge_id, source_keypoint_indices[0], source_keypoint_indices[1]]
    displaced_point = source_keypoint + displacement_vector

#     print("displacements value: ", displaced_point)
#     print("source_keypoint shape: ", source_keypoint.shape)
    
    
#     print("displacements shape: ", displacements.shape)
    
    

    displaced_point_indices = np.clip(
        np.round(displaced_point / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    score = scores[target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]]

    image_coord = displaced_point_indices * output_stride + offsets[
        target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]]
    
    offset = offsets[target_keypoint_id, displaced_point_indices[0], displaced_point_indices[1]]
    # print("--inside traverse keypoint -- *")
    # print("offset shape: ", offset.shape)
    # print("image coord shape: ", image_coord.shape)

    return score, image_coord, displacement_vector, offset

#find the root score and root id and root image coord
def build_part_with_score_torch_single_pose(score_threshold, local_max_radius, scores):
    num_keypoints = scores.shape[0]
    lmd = 2 * local_max_radius + 1
    max_vals = F.max_pool2d(scores, lmd, stride=1, padding=1)

    max_loc = (scores == max_vals) & (scores >= score_threshold)
    max_loc_idx = max_loc.nonzero()

    # Initialize arrays to hold the highest score and corresponding index for each keypoint
    highest_scores = torch.zeros(num_keypoints)
    highest_score_indices = torch.zeros((num_keypoints, 2), dtype=torch.long)

    # Iterate through max_loc_idx and update highest_scores and highest_score_indices
    for idx in max_loc_idx:
        keypoint_idx, y, x = idx
        score = scores[keypoint_idx, y, x]
        if score > highest_scores[keypoint_idx]:
            highest_scores[keypoint_idx] = score
            highest_score_indices[keypoint_idx] = torch.tensor([y, x])
                                      
    return highest_scores, highest_score_indices


def print_decoded_heatmap(heatmap):
    #print heatmap for each image
    os.makedirs('decoded_heatmaps', exist_ok=True)
    
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

#find the root score and root id and root image coord
def find_root(highest_scores, highest_score_indices):
    # Find the index of the keypoint with the highest score
    # print("highest_scores shape: ", highest_scores.shape)
    # print("highest_score_indices shape: ", highest_score_indices.shape)
    # print("highest_score_indices: ", highest_score_indices)
    root_id = torch.argmax(highest_scores).item()
    # print("root_id: ", root_id)

    root_score = highest_scores[root_id].item()



    root_image_coord = highest_score_indices[root_id].cpu().numpy()

    return root_score, root_id, root_image_coord


def decode_pose(
        root_score, root_id, root_image_coord,
        scores,
        offsets,
        output_stride,
        displacements_fwd,
        displacements_bwd
):
    num_parts = scores.shape[0]
    # print("decode pose scores shape: ", scores.shape)
    # print("decode pose num_parts: ", num_parts)
    num_edges = len(PARENT_CHILD_TUPLES)

    instance_keypoint_scores = np.zeros(num_parts)
    instance_keypoint_coords = np.zeros((num_parts, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coord
    
    instance_displacement_vectors = np.zeros((num_edges, 2))
    instance_offsets = np.zeros((num_parts, 2))

    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords, displacement_vector, offset = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_bwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords
            instance_displacement_vectors[edge] = displacement_vector
            instance_offsets[target_keypoint_id] = offset

    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords, displacement_vector, offset = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_fwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords
            instance_displacement_vectors[edge] = displacement_vector
            instance_offsets[target_keypoint_id] = offset
    
    

    return instance_keypoint_scores, instance_keypoint_coords, instance_offsets
