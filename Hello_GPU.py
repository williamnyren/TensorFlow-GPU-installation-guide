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
