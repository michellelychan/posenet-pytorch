#create a Streamlit app using info from image_demo.py
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

@st.cache_data()

def load_model(model):
    model = posenet.load_model(model)
    model = model.cuda()
    return model



def main():
    model_number = st.sidebar.selectbox('Model', [50, 75, 100, 101])
    scale_factor = st.sidebar.slider('Scale Factor', 0.5, 1.5, 1.0)
    image_dir = st.sidebar.text_input('Image Directory', './images_train')
    output_dir = st.sidebar.text_input('Output Directory', './output')

    
    model = load_model(model_number)
    output_stride = model.output_stride

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    filenames = [
        f.path for f in os.scandir(image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    if filenames:
        selected_image = st.selectbox('Choose an image', filenames)
        input_image, draw_image, output_scale = posenet.read_imgfile(
            selected_image, scale_factor=scale_factor, output_stride=output_stride)

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

        if output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

            cv2.imwrite(os.path.join(output_dir, os.path.relpath(selected_image, image_dir)), draw_image)

        st.image(draw_image, caption='PoseNet Output', use_column_width=True)
        st.text("Results for image: %s" % selected_image)

        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            st.text('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                st.text('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))


if __name__ == "__main__":
    main()
