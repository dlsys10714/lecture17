"""CUDA computation backend.

This backend uses cuda backend_ndarray for cached data and redirects
calls to cuda kernel invocations.
"""
from needle import backend_ndarray as nd
from needle.device import Device, DLDeviceType
from . import nd_backend_compute


class CUDADevice(Device):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CUDA, self.device_id)

    def __repr__(self):
        return "cuda(%d)" % self.device_id

    def __str__(self):
        return self.__repr__()

    def array(self, array, dtype):
        return nd.array(array, dtype=dtype, device=nd.cuda())

    def empty(self, shape, dtype):
        return nd.empty(shape, dtype=dtype, device=nd.cuda())

    def to_numpy(self, data):
        return data.numpy()

    def fill(self, array, fill_value):
        array.fill(fill_value)
        return array

    def enabled(self):
        return nd.cuda().enabled()

    def compute(self, op, inputs, attrs):
        """Dispatch device specific computation"""
        # dispatch device specific compute to op.numpy_compute
        # these computation are registered below.
        return op.nd_compute(inputs, attrs)


def cuda(device_id: int = 0) -> CUDADevice:
    return CUDADevice(device_id)
