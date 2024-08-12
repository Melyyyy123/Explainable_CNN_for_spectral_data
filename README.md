# Explainable Convolutional Neural Network for Dating Pine Wood Samples Based on Infrared Spectroscopy

This repository contains the Data, Code and Software versions for my Master's thesis "Convolutional Neural Network for Dating Pine Wood Samples Based on Infrared Spectroscopy".

## Setup
### Virtual environment

The virtual environment was created with Python 3.11.8.

Create the environment (name: tensor):

    virtualenv -p python3.11 tensor

This will create an environment called tensor. Activate it using:

    source tensor/bin/activate

For the installation of the required packages run:  

    pip install -r requirements.txt

## Part 1: Developing an Architecture optimized for Dating Pine Wood Samples Based on their Infrared Spectra

For testing different architectures (raw tuning) and for obtaining the optimal hyperparameters the code from Dário Passos https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra has been used and adapted for my project. 

## Part 2: Explaining the Predictions of the CNNs

In order to explain the predictions of the CNNs the following methods have been used:

1. Occlusion Sensitivity
2. Guided Backpropagation
3. Layer-wise relevance Propagation
4. SHAP Additive Explanations