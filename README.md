# Deep Learning-Based Model for On-Site Earthquake Early Warning in India



## Overview

This document details the implementation of an earthquake early warning system leveraging a variational autoencoder (VAE). The model predicts 27 spectral accelerations based on 8 key ground motion parameters: Peak Ground Acceleration ($PGA$), Peak Ground Displacement ($PGD$), Predominant Frequency ($F_p$), 5-95% Significant Duration ($T_{sig}$), Arias Intensity ($I_a$), Cumulative Absolute Velocity ($CAV$), Site Class ($S_c$), and Direction Flag ($dir$).

##Library Installation and Importing Dependencies

##Installation

Ensure you have Python 3.x installed. Use the command below to install the required dependencies:


pip install numpy pandas matplotlib scikit-learn keras tensorflow```

### Importing Libraries  

The model utilizes various libraries for numerical operations, data handling, visualization, and deep learning.  

## About the Libraries  

- **numpy**: Handles numerical computations and array operations.  
- **pandas**: Manages and processes structured data.  
- **matplotlib.pyplot**: Generates plots and visualizations.  
- **scikit-learn**: Used for data preprocessing and model evaluation.  
- **keras & tensorflow**: Builds and trains deep learning models.  

## Model Architecture  

The system is built on a **Conditional Variational Autoencoder (CVAE)**, which is designed to reconstruct spectral accelerations from ground motion parameters.  

### Key Components  

- **Encoder**: Compresses spectral acceleration ($S_a(T)$) data into a latent representation.  
- **Mapping Layers**: Integrates ground motion parameters (GMPs) into the latent space.  
- **Decoder**: Reconstructs spectral accelerations using the latent representation.  

### Hyperparameters  

- **Network Structure**: Symmetric encoder-decoder setup with hidden layers.  
- **Latent Space**: Dimension set to three ($z_1, z_2, z_3$).  
- **Mapping Layers**: Two hidden layers (4 and 3 nodes), activated with ReLU.  

### Loss Function  

The training objective minimizes a combination of reconstruction and regularization losses:  

- **Reconstruction Loss ($L_{recon}$)**: Measures how well the model reconstructs $S_a(T)$.  
- **Regularization Loss ($L_{reg}$)**: Ensures latent space follows a Gaussian distribution.  

### Model Training Strategy  

- **Training Approach**: Uses K-fold Cross Validation and Adam optimizer.  
- **Initial Training**: VAE is trained for 50 epochs (batch size: 16).  
- **Mapping Network**: Further optimized for 600 epochs (batch size: 64) to enhance input-output relationships.  

![image](https://github.com/PavanMohanN/EEW_system_Variational/assets/65588614/0cb249d7-d8ba-4903-9195-d13aa7cce51a)  

**Fig. 1. Comparison between predicted and true spectral acceleration values.**  

### Sensitivity Analysis  

- **Objective**: Assess model response to ground motion parameters (GMPs).  
- **Visualization**: Latent variables ($z_1, z_2, z_3$) mapped against GMPs to validate model reliability.  
- **Implementation**: The sensitivity analysis procedure is detailed in *Mapping2Output.ipynb*.  

![image](https://github.com/PavanMohanN/EEW_system_Variational/assets/65588614/9f1ca449-893b-4a17-a289-038fb3b17f9f)  

**Fig. 2. Sensitivity analysis for different ground motion parameters.**  

## Input Mapping Layers - The Game Changer  

Since spectral accelerations are unknown in real-time earthquake warning scenarios, the model introduces mapping layers that directly translate input variables to the latent space.  

### Key Features  

- **Dedicated Mapping Layers**: Two separate layers transform the 8 input parameters ($PGA, PGD, F_p, T_{sig}, {I_a}, CAV, S_c, dir$) into a latent space representation.  
- **Concatenation Strategy**: The encoder is disconnected, and mapping layers directly integrate with the decoder.  

- Further details on mapping layers and concatenation can be found in *Concatenation2Compact.ipynb*.  

## Model Output  

This enhanced architecture enables direct prediction of spectral accelerations from ground motion parameters, allowing for **real-time earthquake early warning predictions**.  
