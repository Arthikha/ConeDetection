# ConeDetection

This repository contains a Python script (`coneDetectCNN.py`) to train a Convolutional Neural Network (CNN) for detecting blue and yellow cones in single images or video frames, using the Formula Student Cones Detection IRT dataset from Kaggle. The model predicts 2D bounding boxes and class labels.

## Project Overview

### Concept Explanation
- **Object Detection**: Ientify and localize objects (cones) in images using a CNN. This is achieved by predicting bounding boxes and class labels ("blue" or "yellow") with confidence scores.
- **Convolutional Neural Network (CNN)**: A type of deep learning model that uses convolutional layers to extract spatial features from images, making it ideal for vision tasks like cone detection.

### Component Justifications
- **CNN**: Includes three convolutional layers with ReLU activation and max pooling for feature extraction, followed by fully connected layers for box and class prediction.
- **PyTorch**: Selected for its flexibility in building custom models.

### Methodology
1. **Data Preparation**: Images and YOLO-format labels are loaded from the dataset (given as "formula_student_cones_detection_irt.zip"), resized to 640x640, and normalized. The dataset is processed in batches using a custom `ConeDataset` class.
2. **Model Training**: The CNN is trained for 20 epochs using Adam optimization. Loss combines bounding box regression and classification.
3. **Evaluation**: mAP@0.5 is calculated using `sklearn.metrics.average_precision_score` to assess detection performance.
4. **Inference**: The trained model predicts boxes and labels on new images.

## Source Code, Datasets, and Configurations

### Source Code
- **File**: `coneDetectCNN.py`
  - Contains the implementation, including dataset loading, model definition, training loop, evaluation, and inference.
  - Dependencies: `torch`, `torchvision`, `numpy`, `PIL`, `sklearn.metrics`.

### Datasets
- **Source**: [Formula Student Cones Detection IRT](https://www.kaggle.com/datasets/mfclabber/formula-student-cones-detection-irt)

### Configurations
- **Image Size**: 640x640 pixels.
- **Batch Size**: 2.
- **Max Boxes**: 10 per image.
- **Epochs**: 20.
- **Model**: `cone_detector.pth`.

## Usage of code

1. **Setup**:
   - Download the zip folder and extract the dataset into project folder.
   - Install dependencies: `!pip install torch torchvision numpy sklearn`.

2. **Training**:
   - `python coneDetectCNN.py`

3. **Inference**:
   - Replace `test_image.jpg` with the test image path in the inference section to see predictions.

## References
[1] M. Clabber, "Formula Student Cones Detection IRT," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/mfclabber/formula-student-cones-detection-irt. [Accessed: Oct. 22, 2025].

[2] PyTorch Team, "PyTorch," 2016. [Online]. Available: https://pytorch.org/. [Accessed: Oct. 22, 2025].

[3] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011. [Online]. Available: https://scikit-learn.org/stable/. [Accessed: Oct. 22, 2025].
