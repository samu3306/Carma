import pytest

from scripts.train_dinov2_angle_classifier import resolve_device


class FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self):
        return self._available


class FakeTorch:
    def __init__(self, available: bool):
        self.cuda = FakeCuda(available)


def test_auto_device_prefers_cuda_when_available():
    assert resolve_device(FakeTorch(True), "auto") == "cuda"
    assert resolve_device(FakeTorch(False), "auto") == "cpu"


def test_explicit_unavailable_cuda_is_rejected():
    with pytest.raises(RuntimeError, match="無法使用 CUDA"):
        resolve_device(FakeTorch(False), "cuda")
