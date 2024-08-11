# OCR-from-scratch_-Neural-Networks
OCR From Scratch on MNIST Dataset
Hey everyone! This project is a basic version of Optical Character Recognition (OCR). It's built on the previously uploaded repository Neural Network from Scratch (nnfs) which is a practical implementation of Neural Networks to identify handwritten numbers from the well-known MNIST Dataset.

## Overview
The project demonstrates the implementation of Neural Networks from Scratch for classifying handwritten digits using Forward Propagation, Backward Propagation, Activation Function, Loss Function, and Optimization methods. The dataset utilized is the MNIST Dataset.

## Dataset
The well-known MNIST dataset was created in 1994 to train Artificial Neural Networks. This dataset comprises 60,000 training-labelled images and 10,000 testing-labelled images of handwritten digits, each 28 
×
 28 pixels. These images are classified into 10 classes ranging from 0 to 9.

## Model Architecture
The neural network architecture used for this project consists of a fully connected network.

Input Layer: 128 neurons gets 784 inputs, with ReLU activaton function.
784 inputs are the vector format of the image. since the image is 28 
×
 28 image.shape() gives (28, 28) pixels in dimentions. when vectorised image.shape() gives (1, 784).
Hidden Layer-1: 64 neurons gets 128 inputs, with ReLU activaton function.
Hidden Layer-2: 64 neurons gets 64 inputs, with ReLU activaton function.
Output Layer: 10 neurons gets 64 inputs, with SoftMax activaton function

## Emphasis on Adam Optimizier
The previous optimizer used for the NNFS repository was SGD. The issue with using SGD was the slow learning process and high computational requirements. Switching to Adam (Adaptive Moment Estimation) has made the process faster and lighter in terms of resource consumption. This optimizer operates based on the concept of velocity, adaptively adjusting the step size to find the local minima.

## Special thanks to - 

[Aditya Dikonda] (https://github.com/Adityadikonda10)

