
---

# TensorFlow-GPU Installation Guide

This guide provides a step-by-step process for setting up TensorFlow to leverage NVIDIA GPUs on Ubuntu-based systems. By following this guide, you will ensure that TensorFlow is correctly configured with CUDA and cuDNN libraries for GPU acceleration. The steps include verifying NVIDIA driver and CUDA installations, setting up cuDNN, and configuring a Python environment for TensorFlow.

---

## NVIDIA Driver and CUDA Setup

This guide assumes you already have NVIDIA drivers and CUDA packages installed (version 11.x). Follow these steps to verify your installation:

### Check CUDA Version
To check your CUDA version, run:

```bash
nvcc --version
```

Sample output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.
```

### Check NVIDIA Driver and CUDA Version with NVIDIA-SMI
To check your NVIDIA driver version and the CUDA version it supports, run:

```bash
nvidia-smi
```

Sample output:
```
NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4
```

---

## Install cuDNN Library

The NVIDIA CUDA Deep Neural Network (cuDNN) library provides GPU-accelerated primitives for deep learning. For CUDA 11.x, install the compatible cuDNN version (**v8.9.7** as of December 2023).

### Step 1: Download cuDNN
Download the cuDNN package from the [NVIDIA cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive). Select the package for Ubuntu 22.04 and CUDA 11.x.

### Step 2: Add cuDNN to Package Manager
Register the cuDNN local repository with `dpkg`:

```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
```

### Step 3: Update the Package List
Update the package list to reflect the newly added repository:

```bash
sudo apt-get update
```

### Step 4: Install cuDNN Libraries
1. Install the cuDNN Runtime Library, required for running applications:
   ```bash
   sudo apt-get install libcudnn8=8.9.7.29-1+cuda11.8
   ```

2. Install the cuDNN Developer Library for building applications:
   ```bash
   sudo apt-get install libcudnn8-dev=8.9.7.29-1+cuda11.8
   ```

---

## Python Environment Setup

### Step 1: Install Anaconda
If Anaconda is not installed, [download and install Anaconda](https://www.anaconda.com/products/distribution).

### Step 2: Create a Conda Environment
Create a new isolated environment to manage TensorFlow dependencies:

```bash
conda create --name tensorflow_gpu python=3.11
```

### Step 3: Activate the Environment
Activate the newly created environment:

```bash
conda activate tensorflow_gpu
```

---

## TensorFlow Installation

Install a version of TensorFlow compatible with your CUDA and cuDNN setup. For CUDA 11.x and cuDNN 8.9.x, install TensorFlow **2.14.1**:

```bash
pip install tensorflow==2.15.1
```

> **Note:** TensorFlow recommends using `pip` for installation rather than `conda`. Refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install) for details.

---

## Verifying Installation

To ensure that TensorFlow is properly configured to use your GPU, run the following script:

```python
import tensorflow as tf
from tensorflow.python.client import device_lib

# Check available devices
print("Available devices:")
for device in device_lib.list_local_devices():
    print(f"- {device.name} ({device.device_type})")

# Check if GPU is available
print("\nTensorFlow GPU Support:")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"Num GPUs Available: {len(gpu_devices)}")
    print("GPU Details:")
    for gpu in gpu_devices:
        print(f"  - {gpu.name}")
else:
    print("No GPU detected. TensorFlow is running on CPU.")

# Display TensorFlow version and build details
print("\nTensorFlow Version:", tf.__version__)
print("CUDA Enabled:", tf.test.is_built_with_cuda())
print("cuDNN Enabled:", tf.test.is_built_with_gpu_support())
```

This script provides:
- A list of available devices (CPU and GPU).
- Details on GPU detection and support.
- TensorFlow version and build configuration.

---

This guide ensures a reliable setup of TensorFlow for GPU-based deep learning on Ubuntu systems. Refer to official documentation for updates and additional troubleshooting steps.
