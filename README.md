# Vision-MotionDetect-Evaluation

A computer vision-based 3D human motion detection and evaluation system that utilizes MediaPipe for pose estimation and RNN (Recurrent Neural Network) for intelligent motion assessment. This system can analyze human movements from standard videos and provide professional evaluation feedback.

## ✨ Key Features

• 3D Motion Capture: Extract 3D human pose data from regular 2D videos using MediaPipe

• Intelligent Motion Evaluation: Employ RNN-based deep learning models for professional movement assessment

• Real-time Analysis: Capable of processing video streams in real-time or analyzing pre-recorded footage

• Modular Architecture: Well-structured codebase with separate functional modules for easy customization

• Comprehensive Feedback: Generate detailed evaluation reports on movement quality and form

## 🛠️ Prerequisites & Installation

Python Environment

• Python 3.8 or higher

• pip package manager

Install Dependencies

Install required packages
pip install -r requirements.txt


Or install manually:
Core computer vision and machine learning
pip install opencv-python mediapipe tensorflow keras numpy scipy

Data processing and visualization
pip install matplotlib pandas scikit-learn

Utilities
pip install tqdm pillow


## 🚀 Quick Start

1. Clone the Repository

git clone https://github.com/Zark-byte/Vision-MotionDetect-Evaluation.git
cd Vision-MotionDetect-Evaluation


2. Run the Complete Pipeline (Recommended for First-Time Users)

python runMain.py

This will process the example videos and demonstrate the full motion evaluation workflow.

3. Step-by-Step Execution (For Development and Customization)

Step 1: 3D Pose Capture
python 3Dcapture.py --input examples/example1.mp4 --output output_pose.npy


Step 2: Run Motion Evaluation with RNN Model
python runModule.py --pose_data output_pose.npy --model model/rnn_model.h5


Step 3: Comprehensive Analysis (Main Pipeline)
python main.py --video examples/example1.mp4 --evaluate --visualize


## 🔧 Usage Examples

Basic Video Analysis

python main.py --video path/to/your/video.mp4


Real-time Webcam Analysis

python main.py --webcam --real_time


Batch Processing Multiple Videos

python main.py --batch --input_dir videos/ --output_dir results/


Advanced Options

python main.py --video exercise.mp4 --confidence 0.7 --smooth_landmarks --min_detection_confidence 0.5


## 🧠 Technical Architecture

1. 3Dcapture.py - Pose Estimation Module

• Input: Video file or real-time stream

• Technology: MediaPipe Holistic for 2D-to-3D pose estimation

• Output: 3D skeletal data with 33 * (x, y, z, visibility) coordinates per frame

• Features: Landmark smoothing, coordinate normalization, temporal consistency

2. runModule.py - RNN Evaluation Engine

• Model Type: LSTM/GRU-based recurrent neural network

• Input Features: Temporal sequences of 3D pose data

• Output: Motion quality scores, form assessment, error detection

• Capabilities: 

  • Exercise form evaluation

  • Movement symmetry analysis

  • Range of motion assessment

  • Professional coaching feedback generation

3. main.py - Main Controller

• Orchestrates the complete workflow: Capture → Process → Evaluate → Visualize

• Handles I/O operations and user interface

• Manages configuration parameters and result presentation

## 📊 Output and Results

The system generates comprehensive evaluation reports including:

• Numerical Scores: Overall movement quality (0-100 scale)

• Detailed Feedback: Specific form corrections and suggestions

• Visual Analytics: Side-by-side comparison with ideal form

• Progress Tracking: Historical performance data (when analyzing multiple sessions)

## 🧪 Testing and Validation

Run the test suite to verify installation and basic functionality:
python TEST.py


For model performance testing:
python TEST.py --model_validation --test_set test_videos/


## 🔬 Customization and Extension

Adding New Exercise Models

1. Collect training data for the new exercise
2. Retrain the RNN model with additional classes
3. Update the model configuration in config/exercises.json

Modifying Evaluation Criteria

Edit the assessment parameters in the evaluation module to match specific professional standards or personal requirements.

## 🤝 Contributing

We welcome contributions! Please see our CONTRIBUTING.md for details.

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 📞 Contact and Support

• Project Maintainer: Zark-byte

• Email: chenkexin326@qq.com

For bug reports and feature requests, welcome to communicate with me.

## 🙏 Acknowledgments

• https://mediapipe.dev/ for the robust pose estimation pipeline

• TensorFlow/Keras team for the deep learning framework

• Contributors and testers who helped improve this project
