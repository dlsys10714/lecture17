"""Operator table."""
# Global operator table.
import numpy as np
from numbers import Number
from .autograd import Op, Tensor, Tuple, Value
from .device import default_device
from typing import Optional, List

OP_TABLE = {}


def register_op(name: str, op: Op) -> Op:
    """Register an operator to the op table.

    Parameters
    ----------
    name : str
        The name of the op.

    Returns
    -------
    op : Op
        The registered op.
    """
    if name in OP_TABLE:
        raise ValueError("Op %s is already registered")
    OP_TABLE[name] = op
    return op


def register_op_attr(op_name, attr_name, attr_value=None):
    """Register additional attributes to an existing op by name.


    Parameters
    ----------
    op_name : str
        The name of the op

    attr_name : str
        The name of the attribute

    attr_value :
        The attribute value to be set.

    Returns
    -------
    The attr_value if attr_value is not None.
    Otherwise returns a decorator function.


    Note
    ----
    This function can be used to register additional attributes
    to an Op used by a specific backend.
    """

    def _register(value):
        if op_name not in OP_TABLE:
            raise ValueError("Op %s does not exist")
        op = OP_TABLE[op_name]
        setattr(op, attr_name, value)
        return op

    if attr_value is None:
        return _register
    return _register(attr_value)


class MakeTupleOp(Op):
    def __call__(self, *args: List[Value]) -> Tuple:
        return Tuple.make_from_op(self, list(args))

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, Tuple)
        return [out_grad[i] for i in range(len(out_grad))]


make_tuple = register_op("MakeTuple", MakeTupleOp())


class TupleGetItemOp(Op):
    def __call__(self, a: Tuple, index: int, *, fold_const=True) -> Tensor:
        assert isinstance(a, Tuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTupleOp):
            return a.inputs[index]
        return Tensor.make_from_op(self, [a], attrs={"index": index})

    def gradient(self, out_grad, node):
        index = node.attrs["index"]
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return [make_tuple(*in_grad)]


tuple_get_item = register_op("TupleGetItem", TupleGetItemOp())


class FusedAddScalarsOp(Op):
    def __call__(self, a: Tensor, c0: float, c1: float) -> Tuple:
        return Tuple.make_from_op(self, [a], attrs={"c0": c0, "c1": c1})

    def gradient(self, out_grad, node):
        return [out_grad[0] + out_grad[1]]


fused_add_scalars = register_op("FusedAddScalars", FusedAddScalarsOp())


class EWiseAddOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        return [out_grad, out_grad]


add = register_op("EWiseAdd", EWiseAddOp())


class AddScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad]


add_scalar = register_op("AddScalar", AddScalarOp())


class EWiseMulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs)


multiply = register_op("EWiseMul", EWiseMulOp())


class MulScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad * node.attrs["scalar"]]


multiply_scalar = register_op("MulScalar", MulScalarOp())


class EWiseDivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


divide = register_op("EWiseDiv", EWiseDivOp())


class DivScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


divide_scalar = register_op("DivScalar", DivScalarOp())


class MatMulOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


matmul = register_op("MatMul", MatMulOp())


class SummationOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


summation = register_op("Summation", SummationOp())


class BroadcastToOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


broadcast_to = register_op("BroadcastTo", BroadcastToOp())


class ReshapeOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


reshape = register_op("Reshape", ReshapeOp())


class NegateOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


negate = register_op("Negate", NegateOp())


class TransposeOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


transpose = register_op("Transpose", TransposeOp())


class LogOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


log = register_op("Log", LogOp())


class ExpOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [exp(node.inputs[0]) * out_grad]


exp = register_op("Exp", ExpOp())


class ReLUOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


relu = register_op("ReLU", ReLUOp())


# additional helper functions


def full(shape, fill_value, *, dtype="float32", device=None, requires_grad=False):
    device = device if device else default_device()
    arr = device.empty(shape, dtype)
    device.fill(arr, fill_value)

    return Tensor.make_const(arr, device, requires_grad=requires_grad)


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
