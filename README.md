# TensorFlow-GPU-installation-guide
A installation guide of how to get TensorFlow to use Nvidia GPUs on Ubuntu based systems

### NVIDIA Driver and CUDA Setup Guide

This guide assumes you already have NVIDIA drivers and CUDA packages installed, specifically version 11.x. You can verify your versions using the following commands:

#### Check CUDA Version
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

#### Check NVIDIA Driver and CUDA Version with NVIDIA-SMI
To check your NVIDIA driver version and the CUDA version it supports, run:

```bash
nvidia-smi
```

Sample output:
```
NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4
```

### Install cuDNN Library

The NVIDIA CUDA Deep Neural Network (cuDNN) library provides GPU-accelerated primitives for deep learning. You must download and install the appropriate version for your setup. For CUDA 11.x, the compatible cuDNN version is **v8.9.7** (as of December 5th, 2023). 

#### Step 1: Download cuDNN
Download the cuDNN package from the [NVIDIA cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive). Choose the package for Ubuntu 22.04 and CUDA 11.x.

#### Step 2: Add cuDNN to Package Manager
Use the following command to register the cuDNN local repository with `dpkg` so your system knows where to fetch cuDNN-related packages:

```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
```

#### Step 3: Update the Package List
Run the update command to refresh your package manager’s index:

```bash
sudo apt-get update
```

#### Step 4: Install cuDNN Libraries
1. Install the cuDNN Runtime Library, which is required for running applications using cuDNN:
   ```bash
   sudo apt-get install libcudnn8=8.9.7.29-1+cuda11.8
   ```

2. Install the cuDNN Developer Library, which includes headers and static libraries needed for building applications:
   ```bash
   sudo apt-get install libcudnn8-dev=8.9.7.29-1+cuda11.8
   ```

---

### Python Environment Setup

#### Step 1: Install Anaconda
If Anaconda is not installed, [download and install Anaconda](https://www.anaconda.com/products/distribution) before proceeding. Anaconda simplifies managing Python environments and packages.

#### Step 2: Create a Conda Environment
Create a new isolated environment for TensorFlow to avoid conflicts with other installations:

```bash
conda create --name tensorflow_gpu python=3.11
```

#### Step 3: Activate the Environment
Activate the newly created environment:

```bash
conda activate tensorflow_gpu
```

---

### TensorFlow Installation

TensorFlow requires a specific version compatible with your CUDA and cuDNN setup. For CUDA 11.x and cuDNN 8.9.x, install TensorFlow **2.14.1**:

```bash
pip install tensorflow==2.14.1
```

> **Note:** TensorFlow recommends using `pip` for installation rather than `conda`. Refer to the official TensorFlow installation guide for details: [TensorFlow Installation Guide](https://www.tensorflow.org/install).

---

### Verifying Installation

After completing the installation, you can verify that TensorFlow is using your GPU by running the following Python script:

```python
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

If everything is set up correctly, you should see the number of GPUs detected by TensorFlow.

---

This guide provides a structured approach to setting up CUDA, cuDNN, and TensorFlow for GPU-based deep learning. Always refer to the official documentation for the most up-to-date information.
