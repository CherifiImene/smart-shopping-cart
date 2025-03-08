# Smart Shopping Cart
## Overview

The **Smart Shopping Cart** is  an automatic self checkout system for the automatic segmentation/detection of products in a shopping cart, basket, or checkout counter. 

## Features

- By recognizing individual items from an image or video stream, the system can:

- Identify each product without needing bar codes.
- Speed up the checkout process using AI-powered image recognition.
- Reduce manual errors in product detection and pricing.
- Improve self-checkout efficiency in stores.

## How It Works

1. Camera captures an image of products on a conveyor belt or in a shopping cart.
2. Instance segmentation model separates each product from the background.
3. Recognition module identifies the segmented products using a trained classifier.
4. Total price is calculated, and the customer can pay instantly without manual scanning.

## Dependencies

This application depends on the following Python packages:

- `opencv-python`: Used for handling video streams, drawing bounding boxes, and displaying results.
- `numpy`: Used for numerical computations, including Kalman filter operations.
- `scipy`: Used for calculating distances between tracked objects and detections.
- `matplotlib`: Used for plotting the estimated variables
- `ultralytics`: Used for detecting the objects using yolo11. 

To install the required dependencies, run:

```bash
# create a virtual environment
python -m venv .tracker
# install dependencies
pip install -r requirements.txt
````

## Running the app
```bash
# show all possible options
python track.py --help

python track.py --mode single --video-source 0
````

## System Design
![ System Design](https://github.com/CherifiImene/kalman-object-tracker/blob/main/docs/system_design.png)


## Justification of choices made

### Model : YOLO11s
- Light-weight and fast
- Can be used for real-time applications like object tracking
### Dataset
- Provides pre-trained yolo models with the possibility to deploy them to multiple platform.
- YOLO exported on ONNx format gives x3 speed in cpu.