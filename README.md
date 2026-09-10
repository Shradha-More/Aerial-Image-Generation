# Aerial Image Restoration & 3D Reconstruction

An AI-powered computer vision pipeline for restoring degraded and
partially acquired aerial/UAV imagery and reconstructing useful 3D scene
information from it.

## Overview

Aerial images captured by UAVs can suffer from motion blur, limited
transmission bandwidth, partial image acquisition, and difficult
operating conditions. This project explores a pipeline that combines
image restoration, depth estimation, metric-scale recovery, and 3D
reconstruction to recover meaningful structural information from aerial
imagery.

The system is designed with an emphasis on **offline processing and
deployment in GPS-denied environments**, making it suitable for
scenarios where connectivity and external positioning services may not
be available.

## Key Objectives

-   Restore degraded aerial/UAV images.
-   Estimate and compensate for motion-related blur using available
    telemetry.
-   Reduce image transmission overhead by working with partial image
    information.
-   Estimate scene depth from restored imagery.
-   Convert estimated depth into metric-scale 3D information.
-   Generate point clouds and 3D meshes.
-   Support offline/GPS-denied operation.
-   Optimize inference for resource-constrained deployment.

## Pipeline

``` text
Aerial / UAV Image
        |
        v
Image Acquisition & Preprocessing
        |
        v
PSF Estimation from Telemetry
(speed, altitude, shutter, orientation)
        |
        v
Image Deblurring / Restoration
        |
        v
Depth Estimation
(MiDaS / DPT)
        |
        v
Metric Scale Estimation
        |
        v
3D Point Cloud Generation
(Open3D)
        |
        v
3D Reconstruction
(PIFuHD / DeepSDF)
        |
        v
Offline Visualization / Output
```

## Major Components

### 1. Partial Image Acquisition

Instead of relying on complete image transmission, the system explores
partial acquisition of aerial imagery to reduce communication overhead
while retaining useful scene information.

### 2. PSF-Based Image Deblurring

Motion blur is modeled using a Point Spread Function (PSF). UAV
telemetry such as:

-   Flight speed
-   Altitude
-   Shutter characteristics
-   Camera orientation

is used to estimate blur characteristics and support image restoration.

### 3. Depth Estimation

Deep monocular depth-estimation models such as **MiDaS** and **DPT** are
used to estimate relative scene depth from restored aerial imagery.

### 4. Metric-Scale Recovery

Relative depth is converted toward a useful metric representation using
available scene/camera information, enabling downstream 3D
reconstruction.

### 5. 3D Reconstruction

The estimated depth information is converted into 3D geometry:

-   Point-cloud generation using **Open3D**
-   Surface/3D reconstruction experiments using **PIFuHD** and
    **DeepSDF**

### 6. Offline / GPS-Denied Processing

The pipeline is designed to work without depending on continuous GPS or
cloud connectivity, supporting applications where external positioning
or network access may be unavailable.

## Performance Highlights

The project achieved the following results during experimentation:

  Metric                                                Result
  ------------------------------------- ----------------------
  UAV transmission overhead reduction                \~60--65%
  Reconstruction accuracy                            \~90--95%
  Partial image acquisition                          \~30--35%
  Deployment focus                        Offline / GPS-denied

> **Note:** The reported values are project-specific experimental
> results and depend on the dataset, scene conditions, evaluation
> methodology, and configuration used.

## Datasets / Evaluation Sources

The project experiments and evaluation workflow can use aerial and UAV
datasets such as:

-   **DOTA** --- aerial object detection imagery
-   **UAVDT** --- UAV-based detection and tracking data
-   **Stanford Drone Dataset**
-   **AirSim** --- simulation environment for UAV/robotics experiments

## Model & Deployment Optimization

For practical inference, model optimization techniques were explored
using:

-   **ONNX**
-   **TensorRT**
-   **OpenVINO**
-   FP16 inference
-   INT8 inference

These techniques help reduce inference cost and improve deployment
efficiency on supported hardware.

## Technology Stack

### Computer Vision & Deep Learning

-   Python
-   OpenCV
-   PyTorch
-   MiDaS
-   DPT
-   PIFuHD
-   DeepSDF

### 3D Processing

-   Open3D

### Model Optimization

-   ONNX
-   TensorRT
-   OpenVINO
-   FP16 / INT8

### Backend & Interface

-   FastAPI
-   React
-   Docker

## Key Learning Outcomes

This project provided practical experience in:

-   Image restoration and deblurring
-   Computer vision for aerial imagery
-   Depth estimation
-   3D computer vision
-   Point-cloud processing
-   Deep learning model integration
-   Model optimization and inference acceleration
-   Offline AI system design
-   FastAPI and React integration
-   Working with real-world constraints such as bandwidth, blur, and
    limited connectivity

