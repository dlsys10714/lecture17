import numpy as np
import needle as ndl
from needle import backend_ndarray as nd


def test_nd_empty():
    data = nd.empty([1, 2, 3], device=nd.cpu())
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

def test_strides():
    data = nd.array([1, 2, 3, 4], device=nd.numpy_device())
    assert data.strides == (1,)
    y  = data.as_strided((2, 1), (2, 1))
    assert y.shape == (2, 1)
    assert y.numpy()[1, 0] == 3
    assert not y.is_compact()
    z = y.compact()
    assert z.compact()
    np.testing.assert_equal(z.numpy(), y.numpy())


if __name__ == "__main__":
    test_nd_empty()
    test_nd_array()
    test_strides()
