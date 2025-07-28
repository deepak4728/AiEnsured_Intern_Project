# AiEnsured_Intern_Project

# Vegetable Image Classification

This project implements a vegetable image classification model using TensorFlow and PyTorch. The goal is to classify images of 15 different vegetables.

## Dataset

The dataset used for this project is the "Vegetable Image Dataset" from Kaggle. It contains images of 15 different types of vegetables, split into training, validation, and test sets.

- **Kaggle Dataset Link:** [https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

## Project Structure

The notebook is structured as follows:

1.  **Imports and Dataset Fetch:** Imports necessary libraries and downloads the dataset using `kagglehub`.
2.  **Utility Functions:** Contains helper functions for loading data, plotting training history, and plotting confusion matrices.
3.  **Load Dataset:** Loads the dataset into TensorFlow `tf.data.Dataset` objects.
4.  **Dataset Preview:** Displays sample images from each class in the dataset.
5.  **Custom Model Training using TensorFlow:** Defines and trains a custom Convolutional Neural Network (CNN) model using TensorFlow.
6.  **Plots:** Visualizes the training history and confusion matrices for the custom TensorFlow model.
7.  **Predictions:** Includes a function for making predictions on individual images using the trained TensorFlow model and demonstrates its usage.
8.  **ResNet50 Model Implementation using Tensorflow:** Defines and trains a ResNet50 model using TensorFlow for transfer learning.
9. **ResNet50 using PyTorch:** Implements and trains a ResNet50 model using PyTorch.

## Models Implemented

This project explores vegetable image classification using three different approaches:

1.  **Custom CNN Model (TensorFlow):** A simple Convolutional Neural Network built from scratch using TensorFlow. The architecture consists of several convolutional layers with ReLU activation and batch normalization, followed by max pooling layers. The flattened output is then passed through dense layers with L2 regularization and dropout for classification.
[vegetable_model_128x128.keras](https://drive.google.com/file/d/1vdGyQ3s_eoVqVJLPGxTltp8Elzk3iPxM/view?usp=drive_link)

2.  **ResNet50 (TensorFlow):** This approach utilizes the pre-trained ResNet50 model from Keras Applications for transfer learning. The convolutional base of ResNet50 is used, and new dense layers are added on top for classification. The layers of the ResNet50 model are trained to fine-tune the model for this specific dataset.
[vegetable_model_resnet_tf_128x128.keras](https://drive.google.com/file/d/1G6OYNGDlA5BLVYEJwa9d2PuTA58Dd8N6/view?usp=drive_link)

3.  **ResNet50 (PyTorch):** This implementation uses the pre-trained ResNet50 model available in PyTorch's `torchvision.models`. Similar to the TensorFlow transfer learning approach, the final fully connected layer is replaced to match the number of vegetable classes. The model is then trained on the dataset using PyTorch's training loop.
[vegetable_model_resnet_pytorch_128x128.pth](https://drive.google.com/file/d/19uei-97HAPbtCYClBFsJmaKy_a_YUKLo/view?usp=drive_link)

## Getting Started

### Prerequisites

-   Python 3.6+
-   TensorFlow
-   PyTorch
-   Kagglehub
-   Numpy
-   Matplotlib
-   Scikit-learn
-   Seaborn
-   Pillow

You can install the required packages using pip:
