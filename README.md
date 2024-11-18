# TensorFlow-GPU Installation Guide

This guide provides a clear, step-by-step process to set up TensorFlow for NVIDIA GPU acceleration on Ubuntu-based systems. By following this guide, you will configure TensorFlow with CUDA and cuDNN libraries, enabling efficient deep learning workflows.

---

## Prerequisites: NVIDIA Driver and CUDA Setup

Ensure you have NVIDIA drivers and CUDA packages installed (version 11.x). Use the following commands to verify your setup:

### Check CUDA Version

To check the installed CUDA version, run:

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

### Check NVIDIA Driver and GPU Information

To view your NVIDIA driver version, GPU name, and total memory:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

Sample output:

```
name, driver_version, memory.total [MiB]
NVIDIA GeForce GTX 1650, 470.256.02, 3889 MiB
NVIDIA TITAN V, 470.256.02, 12066 MiB
```

---

## Install cuDNN Library

The NVIDIA CUDA Deep Neural Network (cuDNN) library provides GPU-accelerated primitives for deep learning. For CUDA 11.x, the compatible cuDNN version is **v8.9.7**.

### Step 1: Download cuDNN

1. Visit the [NVIDIA cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive).
2. Download the appropriate package for your operating system.

#### Step 1a: Verify Your OS

If unsure about your OS details, install and use `neofetch`:

```bash
sudo apt install neofetch
neofetch --stdout | grep OS:
```

Sample output:

```
OS: Ubuntu 22.04.5 LTS x86_64
```

> **Note:** If you are using Kubuntu, the underlying system is the same as Ubuntu, and the version numbers correspond (e.g., Kubuntu 22.04 is equivalent to Ubuntu 22.04).

---

### Step 2: Add cuDNN to Package Manager

Register the downloaded cuDNN repository with `dpkg`:

```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
```

### Step 3: Update the Package List

Refresh your package manager to include the cuDNN repository:

```bash
sudo apt update
```

### Step 4: Install cuDNN Libraries

1. Install the **Runtime Library** (required for running applications):

   ```bash
   sudo apt-get install libcudnn8=8.9.7.29-1+cuda11.8
   ```

2. Install the **Developer Library** (required for building applications):

   ```bash
   sudo apt-get install libcudnn8-dev=8.9.7.29-1+cuda11.8
   ```

---

## Python Environment Setup

### Step 1: Install Anaconda

If you don’t have Anaconda installed, [download and install Anaconda](https://www.anaconda.com/products/distribution).

### Step 2: Create a Conda Environment

Set up a new isolated environment for TensorFlow:

```bash
conda create --name tensorflow_gpu python=3.11
```

### Step 3: Activate the Environment

Activate the environment to install TensorFlow:

```bash
conda activate tensorflow_gpu
```

---

## TensorFlow Installation

Install a version of TensorFlow compatible with your CUDA and cuDNN setup. For CUDA 11.x and cuDNN 8.9.x, install TensorFlow **2.15.1**:

```bash
pip install tensorflow==2.15.1
```

> **Note:** TensorFlow recommends using `pip` for installation instead of `conda`. For more details, refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install).

---

## Verifying Installation

To confirm that TensorFlow is configured to use your GPU, run the following script:

```python
import tensorflow as tf
from tensorflow.python.client import device_lib

# Display TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Check available devices
print("\nAvailable devices:")
for device in device_lib.list_local_devices():
    print(f"- {device.name} ({device.device_type})")

# Check GPU details
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"\nNum GPUs Available: {len(gpu_devices)}")
    for gpu in gpu_devices:
        print(f"  - {gpu.name}")
else:
    print("\nNo GPU detected. TensorFlow is running on CPU.")

# Check CUDA and cuDNN build
print("\nBuild Information:")
print(f"CUDA Enabled: {tf.test.is_built_with_cuda()}")
print(f"cuDNN Enabled: {tf.test.is_built_with_gpu_support()}")
```

This script outputs:

- TensorFlow version.
- Available devices (CPU and GPU).
- Number and details of GPUs detected.
- Build information for CUDA and cuDNN support.

---
