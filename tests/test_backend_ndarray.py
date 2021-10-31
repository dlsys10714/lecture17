import numpy as np
import needle as ndl
from needle import backend_ndarray as nd


def test_nd_empty():
    data = nd.empty([1, 2, 3], device=nd.cpu())
    print(data.shape)
    assert data.shape == (1, 2, 3)


def test_nd_array():
    def check(nd_device):
        if not nd_device.enabled():
            print("Skip because %s is not enabled" % device)
            return
        np_arr = np.array([1, 2, 3])
        x0 = nd.array(np_arr, device=device)
        x1 = x0 + 1

        np.testing.assert_equal(x0.numpy(), np_arr)
        np.testing.assert_equal(x1.numpy(), np_arr + 1)

    for device in [nd.numpy_device(), nd.cuda()]:
        check(device)


if __name__ == "__main__":
    test_nd_empty()
    test_nd_array()
