"""
src package initialization for CORDEX QRE Reproduction.
"""
from . import gpu_setup  # noqa: F401  (sets TF_FORCE_GPU_ALLOW_GROWTH before TensorFlow uses the GPU)
