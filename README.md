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

```\begin{align*} \text{Compute the biased first moment estimate } m_{t} \\
m_{t} &= \beta_{1} \cdot m_{t-1} + (1 - \beta_{1}) \cdot g_{t} \\
\text{where } g_{t} \text{ is the gradient at time step } t. \\
\text{Compute the biased second raw moment estimate } v_{t} \\
v_{t} &= \beta_{2} \cdot v_{t-1} + (1 - \beta_{2}) \cdot g_{t}^{2} \\
\text{Compute bias-corrected first moment estimate } \hat{m}^{t} \\
\hat{m}_{t} &= \frac{m_{t}}{1 - \beta_{1}^{t}} \\
\text{Compute bias-corrected second raw moment estimate } \hat{v}^{t} \\
m_{t} &= \beta_{1} \cdot m_{t-1} + (1 - \beta_{1}) \cdot g_{t} \\
\text{Update the parameters } \theta \\
\theta_{t} &= \theta_{t-1} - \frac{a \cdot \hat{v}_{t}}{\sqrt{\hat{m}_{t} + \epsilon}}
\end{align*}```

## Training
The file training_OCR.py is used to train and save the model. The training process involves classifying the input data to the correct class through forward propagation and updating the weights and biases of the network after each epoch through backpropagation.

Learning Rate is set to 0.01.

number of epochs is set to 301

Training file contains classes:

Layer_Dense
ActivationReLU
ActivationSoftMax
Loss
LossfunctionLossCategoricalCrossEntropy
Optimizer_Adam
The trained model is saved as OCR_Model_128,64,64,10.pkl.

## Testing
The trained model can be used to test the 10,000 test images from the MNIST Dataset. In the file testing_OCR.py, a random batch of 10 examples is tested in series of 5.

## Results
Loss and Accuracy are tracked at every epoch and plottet for visualisation.

## Perfomance
Highest achieved Accuracy is 99% at 300th epoch.
90% Accuracy achieved at 53rd epoch.
Dependencies
Numpy
Matplotlib
Keras
How to Run
To run the code and train the neural network, follow these steps:

## Clone the Repository:
git clone https://github.com/Adityadikonda10/Neural-Network-from-Scratch-nnfs-
Install Dependencies:
pip install numpy
pip install matplotlib
pip install keras
Training the Model:
python training_OCR.py
Testing the Model:
python testing_OCR.py
Acknowledgements
This project is primarily based on nnfs by harrison Kinsley at Sentdex's YouTube playlist

## Neural Networks from Scratch.
Samson Zhang's video on

## Building a neural network FROM SCRATCH
Andreas Zinonos's video on

Beginner Deep Learning Tutorial | MNIST Digits Classification Neural Network in Python, Keras YouTube video.
CampusX's video on

Adam Optimizer Explained in detail


