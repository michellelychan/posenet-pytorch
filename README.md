# PoseNet Pytorch

This repository contains a PyTorch implementation (multi-pose only) of the Google TensorFlow.js Posenet model (https://github.com/tensorflow/tfjs-models/tree/master/pose-detection). It is built on top of Ross Wightman’s posenet-pytorch implementation ([https://github.com/rwightman/posenet-pytorch](https://github.com/rwightman/posenet-pytorch)), but it also includes the following, which allows you to create your ground truth data and fine-tune the model from end-to-end : 

1. **Train File** (for multi-person loss) 
2. ********************Ground-truth Generation******************** from annotations from Dataloop and Roboflow
3. **Visualize your heatmaps** from Training 
4. **Streamlit Demo App** 
    1. Run the default pre-trained model on images and video. Adjust output stride, pose & keypoint confidence scores to find the optimal parameters for your use case.  
    2. Run on a trained model (WIP) 

Further optimization is possible as the MobileNet base models have a throughput of 200-300 fps. 

# Install

A suitable Python 3.x environment with a recent version of PyTorch is required. 

Development and testing was done with Python 3.9.6 and PyTorch 1.12.1 w/ CUDA 11.6. 

A fresh conda Python 3.9 environment with the following installs should suffice:

`conda install -c pytorch pytorch cudatoolkit
pip install requests opencv-python==4.6.0`

# Demo

There are 4 demo apps in the root that utilize the PoseNet model. They are very basic and could definitely be improved.

The first time these apps are run (or the library is used) model weights will be downloaded from the TensorFlow.js version and converted on the fly.

For all demos, the model can be specified with the '--model` argument by using its integer depth multiplier (50, 75, 100, 101). The default is the 101 model. 

### 1. streamlit_demo.py

Streamlit app interface that allows you to upload an image or video to see the keypoints generated. You can: 

1. Run the default pre-trained model on images and video. Adjust output stride, pose & keypoint confidence scores to find the optimal parameters for your use case.  
2. Run on a trained model (WIP)

`streamlit run streamlit_demo.py`

### 2. image_demo.py

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton overlayed.

`python image_demo.py --model 101 --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the `get_test_images.py` script.

### 3. benchmark.py

A minimal performance benchmark based on image_demo. Images in `--image_dir` are pre-loaded and inference is run `--num_images` times with no drawing and no text output.

### 4. webcam_demo.py

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and skeletons and rendered to the screen. The default args for the webcam_demo assume device_id=0 for the camera and that 1280x720 resolution is possible.

# Training the Model

### ground_truth_dataloop.py (Dataloop.ai)

Generates ground truth heatmaps and offset vectors from keypoints annotated from [Dataloop.ai](http://Dataloop.ai). To create pose estimation annotations, check out: [https://dataloop.ai/docs/create-annotation-point](https://dataloop.ai/docs/create-annotation-point) . Works for both multi and single person. 

### ground_truth_roboflow.py (Roboflow) (WIP)

Generates ground truth heatmaps from keypoints annotated from Roboflow. Note: Roboflow has not developed an annotation feature for specifically for pose estimation at the moment. It is a simpler interface and works well for single-person pose estimation. 

- Sample Dataset: [https://universe.roboflow.com/michelle-chan/human-body-pose-ground-truth](https://universe.roboflow.com/michelle-chan/human-body-pose-ground-truth)

### train.py

Implemented loss functions for multi-person pose estimation based on paper [“Towards Accurate Multi-person Pose Estimation in the Wild”](https://arxiv.org/pdf/1701.01779.pdf). 

Training Charts are created with weights and biases (https://wandb.ai/). To start training, create a wandb.ai account and use your API key. 

# Other Tools

### visualizers.py

Generates Heatmaps for visualization and saves them to an image file. 

# Credits

This work is not related to Google. 

The repo is based off of Ross Wightman’s posenet-pytorch: [https://github.com/rwightman/posenet-python](https://github.com/rwightman/posenet-python) 

The original model, weights, code, etc. was created by Google and can be found at [https://github.com/tensorflow/tfjs-models/tree/master/posenet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) 

They have a newer pose-detection package here: [https://github.com/tensorflow/tfjs-models/tree/master/pose-detection](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection)

The Python conversion code was adapted from the CoreML port at [https://github.com/infocom-tpo/PoseNet-CoreML](https://github.com/infocom-tpo/PoseNet-CoreML) 

# References

Research Papers: 

1. **Towards Accurate Multi-person Pose Estimation in the Wild:** [https://arxiv.org/pdf/1701.01779.pdf](https://arxiv.org/pdf/1701.01779.pdf)
2. **PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model:** 
    
    [https://arxiv.org/pdf/1803.08225.pdf](https://arxiv.org/pdf/1803.08225.pdf) 
    

Articles: 

- ****Real-time Human Pose Estimation in the Browser with TensorFlow.js:****
    
    [https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5](https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5)