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

    model = load_model(model_number)
    output_stride = model.output_stride

    option = st.sidebar.selectbox('Choose an option', ['Upload Image', 'Upload Video', 'Try existing image'])
    output_dir = st.sidebar.text_input('Output Directory', './output')

    if option == 'Upload Video':
        uploaded_video = st.sidebar.file_uploader("Upload a video (mp4, mov, avi)", type=['mp4', 'mov', 'avi'])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
        
            vidcap = cv2.VideoCapture(tfile.name)
            success, image = vidcap.read()
            frames = []
            frame_count = 0
            

            while success:
                # image = process_frame(image, model, output_stride, output_scale, scale_factor)
                input_image, draw_image, output_scale = process_frame(image, scale_factor, output_stride)
                result_image, pose_scores, keypoint_scores, keypoint_coords = run_model(input_image, draw_image, model, output_stride, output_scale)
                # print_frame(result_image, pose_scores, keypoint_scores, keypoint_coords, output_dir, filename)
                
                if result_image is not None:
                    frames.append(result_image)
                    success, image = vidcap.read()
                    frame_count += 1
                else:
                    st.text("Failed to process a frame.")
            
            progress_bar = st.progress(0)
                
            # Write the output video
            output_file = 'output.mp4'
            height, width, layers = frames[0].shape
            size = (width,height)
            out = cv2.VideoWriter(os.path.join(output_dir, output_file), cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

            for i in range(len(frames)):
                progress_percentage = i / len(frames)
                progress_bar.progress(progress_percentage)
                out.write(frames[i])
            
            st.video(video_bytes)
            
            if frames:
                frame_idx = st.slider('Choose a frame', 0, len(frames) - 1, 0)
                st.image(frames[frame_idx], caption=f'Frame {frame_idx + 1}', use_column_width=True)

            progress_bar.progress(1.0)
            out.release()

            video_file = open(output_file, 'rb')
            video_bytes = video_file.read()
            

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

            result_image, pose_scores, keypoint_scores, keypoint_coords = run_model(input_image, source_image, model, output_stride, output_scale, output_dir, filename)
            print_frame(result_image, pose_scores, keypoint_scores, keypoint_coords, output_dir, filename=filename)
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
            print_frame(result_image, pose_scores, keypoint_scores, keypoint_coords, output_dir, filename=selected_image)

        
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
    
def run_model(input_image, draw_image, model, output_stride, output_scale):
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

        draw_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        return draw_image, pose_scores, keypoint_scores, keypoint_coords

def print_frame(draw_image, pose_scores, keypoint_scores, keypoint_coords, output_dir, filename=None):
        if output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)
        
            if filename:
                cv2.imwrite(os.path.join(output_dir, filename), draw_image)
            else:
                cv2.imwrite(os.path.join(output_dir, "output.png"), draw_image)
        
            st.image(draw_image, caption='PoseNet Output', use_column_width=True)
            st.text("Results for image: %s" % filename)

            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                st.text('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    st.text('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

if __name__ == "__main__":
    main()
