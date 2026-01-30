# Convolutional Autoencoders for Image Reconstruction and Colorization

This project investigates the design and evaluation of **Convolutional Autoencoders (CAEs)** for image reconstruction and colorization tasks. Multiple architectures were implemented and compared to analyze the impact of **model depth, filter size, latent space dimensionality, and learning rate** on reconstruction performance.

The project was developed as part of a university assignment focused on representation learning and unsupervised deep learning.

---

## Project Overview

Autoencoders are neural networks trained to learn compact representations of input data by reconstructing the input at the output layer.  
In this project, convolutional autoencoders are applied to image data to:

- Reconstruct grayscale and RGB images
- Analyze reconstruction error using **Mean Squared Error (MSE)**
- Perform image colorization using both **RGB** and **YUV** color spaces

---

## Exercises Breakdown

### Exercise 1 — Autoencoder Implementation
- Dataset split into **training, validation, and test sets**
- Pixel values normalized to the range **[-1, 1]**
- Convolutional Autoencoder implemented using **Keras**
- Training performed for **15 epochs**
- Best baseline model achieved a test MSE of **0.0179**

---

### Exercise 2 — Architecture Comparison
Several CAE architectures were evaluated, varying:
- Number of convolutional filters
- Kernel sizes
- Latent space dimensionality

Key observations:
- Deeper models with more filters achieved **lower reconstruction error**
- No direct linear correlation between latent space size and MSE
- The **32–48–64 single-filter architecture** offered the best trade-off between performance and training time

---

### Exercise 3 — Image Colorization
The best-performing architecture was used for colorization tasks:

- **RGB approach**:  
  - Input: grayscale image (1 channel)  
  - Output: RGB image (3 channels)

- **YUV approach**:  
  - Input: luminance (Y channel)  
  - Output: chrominance (UV channels)

Results:
- RGB colorization MSE: **0.006633**
- YUV colorization MSE: **0.003041**
- YUV approach provided better performance due to separating luminance and chrominance information

---

## Evaluation Metric

- **Mean Squared Error (MSE)** was used to evaluate reconstruction and colorization quality
- Training and validation curves show consistent convergence across experiments

---

## Technologies Used

- Python
- TensorFlow / Keras
- Convolutional Neural Networks
- Autoencoders
- Image processing (RGB & YUV color spaces)

---

## Future Improvements

- Training for more epochs to improve convergence
- Testing intermediate learning rates (e.g. 10⁻⁴)
- Using deeper architectures before downsampling
- Exploring perceptual loss functions instead of MSE

---

## Authors

- **Popa Ștefan-Andrei**
- **Eduard Levinschi**

---

## Documentation

A detailed explanation of the methodology, experiments, and results is available in the project report:
- *Report_Autoencoders_Popa_Stefan_Andrei_Levinschi_Eduard.pdf*
