"""GPU settings shared by every entry point and test.

Importing this module (``src/__init__.py`` does) sets TF_FORCE_GPU_ALLOW_GROWTH=true
unless it is already set, so a TensorFlow process allocates GPU memory as needed
instead of reserving all of it. Without that, a process that already used the GPU
(e.g. pytest running models in-process) leaves none for a child process, whose cuDNN
calls then fail with CUDNN_STATUS_INTERNAL_ERROR / EXECUTION_FAILED. The variable is
read when TensorFlow creates its GPU device, so it only has to be set before the
first TensorFlow operation, and child processes inherit it.

GPU_RTOL: on a GPU, results are not bit-identical between two evaluations (cuDNN may
pick other kernels); checks that are exact on a CPU allow this relative difference.
"""
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

GPU_RTOL = 1e-5


def enable_memory_growth():
    """Also turn on memory growth through tf.config (call before the first TensorFlow op).

    Returns the number of GPUs TensorFlow sees."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # GPU already initialized; TF_FORCE_GPU_ALLOW_GROWTH (set above) applied then
    return len(gpus)
