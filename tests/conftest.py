"""pytest setup shared by all tests.

TF_FORCE_GPU_ALLOW_GROWTH: tests run models in the pytest process and then start
training subprocesses; without memory growth the pytest process reserves all GPU
memory and the subprocess's cuDNN calls fail (CUDNN_STATUS_INTERNAL_ERROR). Set here,
before any test module imports TensorFlow, and inherited by the subprocesses.
"""
import os
import sys

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(TESTS_DIR)
for p in (WORKSPACE_ROOT, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    # slow: integration runs of the post-lab studies (about 10 min on a laptop CPU, minutes on a GPU);
    # the smoke suite skips them (-m "not slow"); run them with  python -m pytest -m slow tests/test_mass_study.py
    config.addinivalue_line("markers", "slow: long integration test, not part of the smoke suite")
