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
    for i, gpu in enumerate(gpu_devices):
        print(f"  - GPU {i}: {gpu.name}")
else:
    print("\nNo GPU detected. TensorFlow is running on CPU.")

# Check CUDA and cuDNN build
print("\nBuild Information:")
print(f"CUDA Enabled: {tf.test.is_built_with_cuda()}")
print(f"cuDNN Enabled: {tf.test.is_built_with_gpu_support()}")

# Dummy computation on each GPU
if len(gpu_devices) >= 2:
    print("\nPerforming dummy computations on GPUs...")

    # Define small tensors
    tensor_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    tensor_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

    # Load tensors to GPU 0
    with tf.device('/GPU:0'):
        gpu0_result = tf.add(tensor_a, tensor_b)
        print(f"\nTensor addition on GPU 0:\n{gpu0_result.numpy()}")

    # Load tensors to GPU 1
    with tf.device('/GPU:1'):
        gpu1_result = tf.multiply(tensor_a, tensor_b)
        print(f"\nTensor multiplication on GPU 1:\n{gpu1_result.numpy()}")
else:
    print("\nAt least two GPUs are required to perform dummy computations.")
