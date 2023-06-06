#create a Streamlit app using info from image_demo.py
import cv2
import time
import argparse
import os
import torch
import posenet
import tempfile
from posenet.utils import *
import streamlit as st
from posenet.decode_multi import *
from visualizers import *
from ground_truth_dataloop import *

import cv2
import time
import argparse
import os
import torch
import posenet
import streamlit as st
from posenet.decode_multi import *
from visualizers import *
from ground_truth_dataloop import *

st.title('PoseNet Image Analyzer')

def process_frame(frame, scale_factor, output_stride):
    input_image, draw_image, output_scale = process_input(frame, scale_factor=scale_factor, output_stride=output_stride)
    return input_image, draw_image, output_scale

@st.cache_data()

def load_model(model):
    model = posenet.load_model(model)
    model = model.cuda()
    return model

def main():
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

    model_number = st.sidebar.selectbox('Model', [101, 100, 75, 50])
    scale_factor = 1.0
    output_stride = st.sidebar.selectbox('Output Stride', [8, 16, 32, 64])
    min_pose_score = st.sidebar.number_input("Minimum Pose Score", min_value=0.000, max_value=1.000, value=0.10, step=0.001)
    st.sidebar.markdown(f'<p style="color:grey; font-size: 12px">The current number is {min_pose_score:.3f}</p>', unsafe_allow_html=True)

    min_part_score = st.sidebar.number_input("Minimum Part Score", min_value=0.000, max_value=1.000, value=0.010, step=0.001)
    st.sidebar.markdown(f'<p style="color:grey; font-size:12px">The current number is {min_part_score:.3f}</p>', unsafe_allow_html=True)

    model = load_model(model_number)
    output_stride = model.output_stride

    option = st.sidebar.selectbox('Choose an option', ['Upload Image', 'Upload Video', 'Try existing image'])
    output_dir = st.sidebar.text_input('Output Directory', './output')

    if option == 'Upload Video':
        video_display_mode = st.sidebar.selectbox("Video Display Mode", ['Frame by Frame', 'Entire Video'])
        uploaded_video = st.sidebar.file_uploader("Upload a video (mp4, mov, avi)", type=['mp4', 'mov', 'avi'])
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
        
            vidcap = cv2.VideoCapture(tfile.name)
            success, image = vidcap.read()
            frames = []
            frames_with_keypoints = []
            frame_count = 0
        
            while success:
                input_image, draw_image, output_scale = process_frame(image, scale_factor, output_stride)
                pose_scores, keypoint_scores, keypoint_coords = run_model(input_image, model, output_stride, output_scale)

                result_image_with_keypoints = print_frame(draw_image, pose_scores, keypoint_scores, keypoint_coords, output_dir, min_part_score=min_part_score, min_pose_score=min_pose_score)

                if result_image_with_keypoints is not None and result_image is not None:
                    
                    frames.append(result_image)
                    frames_with_keypoints.append(result_image_with_keypoints)
                    
                    success, image = vidcap.read()
                    frame_count += 1

            if frames:
                frame_idx = st.slider('Choose a frame', 0, len(frames) - 1, 0)
                input_image, draw_image, output_scale = process_frame(frames[frame_idx], scale_factor, output_stride)
                pose_scores, keypoint_scores, keypoint_coords = run_model(input_image, model, output_stride, output_scale)
            
                # Store pose coordinates for each processed frame
                pose_data = {
                    'pose_scores': pose_scores.tolist(),
                    'keypoint_scores': keypoint_scores.tolist(),
                    'keypoint_coords': keypoint_coords.tolist()
                }

                # Print pose data
                

                if result_image is not None:
                    st.image(draw_image, caption=f'Frame {frame_idx + 1}', use_column_width=True)
                    st.write(pose_data)
                else:
                    st.text("Failed to process the frame.")

    elif option == 'Upload Image':
        image_file = st.sidebar.file_uploader("Upload Image (Max 10MB)", type=['png', 'jpg', 'jpeg'])
        
        if image_file is not None:
            if image_file.size > MAX_FILE_SIZE:
                st.error("File size exceeds the 10MB limit. Please upload a smaller file.")
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, 1)
            filename = image_file.name
            # Crop the image here as needed
            # input_image = input_image[y:y+h, x:x+w]
            
            input_image, source_image, output_scale = process_input(
                input_image, scale_factor, output_stride)

            pose_scores, keypoint_scores, keypoint_coords = run_model(input_image, model, output_stride, output_scale)
            print_frame(source_image, pose_scores, keypoint_scores, keypoint_coords, output_dir, filename=filename, min_part_score=min_part_score, min_pose_score=min_pose_score)
        else:
            st.sidebar.warning("Please upload an image.")

   
    
    elif option == 'Try existing image':
        image_dir = st.sidebar.text_input('Image Directory', './images_train')

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        filenames = [f.path for f in os.scandir(image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        if filenames:
            selected_image = st.sidebar.selectbox('Choose an image', filenames)

            input_image, draw_image, output_scale = posenet.read_imgfile(
                selected_image, scale_factor=scale_factor, output_stride=output_stride)

            filename = os.path.basename(selected_image)
            result_image, pose_scores, keypoint_scores, keypoint_coords = run_model(input_image, draw_image, model, output_stride, output_scale)
            print_frame(result_image, pose_scores, keypoint_scores, keypoint_coords, output_dir, filename=selected_image, min_part_score=min_part_score, min_pose_score=min_pose_score)

        
    else:
        st.sidebar.warning("No images found in directory.")    

#same as utils.py _process_input
def process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = posenet.valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])
    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, source_img, scale
    
def run_model(input_image, model, output_stride, output_scale):

    with torch.no_grad():
        input_image = torch.Tensor(input_image).cuda()

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
            
        # st.text("model heatmaps_result shape: {}".format(heatmaps_result.shape))
        # st.text("model offsets_result shape: {}".format(offsets_result.shape))

        pose_scores, keypoint_scores, keypoint_coords, pose_offsets = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.0)

        # st.text("decoded pose_scores shape: {}".format(pose_scores.shape))
        # st.text("decoded pose_offsets shape: {}".format(pose_offsets.shape))

        keypoint_coords *= output_scale

          # Convert BGR to RGB
        
        return pose_scores, keypoint_scores, keypoint_coords

def print_frame(draw_image, pose_scores, keypoint_scores, keypoint_coords, output_dir, filename=None, min_part_score=0.01, min_pose_score=0.1):
        if output_dir:
            
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=min_pose_score, min_part_score=min_part_score)
        
            draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)

            if filename:
                cv2.imwrite(os.path.join(output_dir, filename), draw_image)
            else:
                cv2.imwrite(os.path.join(output_dir, "output.png"), draw_image)
        
            st.image(draw_image, caption='PoseNet Output', use_column_width=True)
            st.text("Results for image: %s" % filename)
            st.text("Size of draw_image: {}".format(draw_image.shape))

            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                st.text('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    st.text('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

if __name__ == "__main__":
    main()