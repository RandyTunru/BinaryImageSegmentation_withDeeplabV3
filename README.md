# Binary Image Segmentation with DeepLabv3-ResNet50

## Introduction

This project implements a binary segmentation to differentiate foreground and background objects from an image, using the pre-trained DeepLabv3 from torchvision, I substituted to final layer for a single output channel and then proceed to fine-tune the model by freezing the backbone layers and only updating the parameters in the classifier layer. I've also developed a downstream application using the output mask of the model to remove background from the image and get the foreground object of the image in png format, which is commonly used in creative aspects such as editing.

## Main Deliverables

- Training code of the DeepLabV3 for Binary Segmentation
- Inference code with Downstream Application for background removal of an image
- Model state (in .pth) can be accessed here: [Gdrive](https://drive.google.com/drive/folders/1un3hQg6ev3cqUaQTZ6eKmm73G8s6mi3g?usp=sharing)

## Example Output