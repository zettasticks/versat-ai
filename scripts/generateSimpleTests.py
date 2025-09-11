import onnx
from onnx import TensorProto, version_converter
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model
from skl2onnx.helpers.onnx_helper import save_onnx_model
from onnx import numpy_helper
from dataclasses import dataclass
from onnxOperators import BroadCastShape, ExtendShape

import sys
import numpy as np
import onnxruntime as ort
import os
import shutil


@dataclass
class Test:
    shapes: list[list[int]] = None
    tensors: list[any] = None
    outputTensor: any = None
    node: any = None
    randomArrays: list[any] = None
    initializerArrays: list[any] = None


tests: list[Test] = []


def GetInputTrueName(testIndex, inputIndex):
    VARS = ["X", "Y", "Z"]
    assert inputIndex < len(VARS)

    return f"{VARS[inputIndex]}{testIndex}"


def GetOutputTrueName(testIndex):
    return f"OUT{testIndex}"


def GetInitializerTrueName(testIndex):
    return f"A{testIndex}"


def CreateBinaryOpTest(op, leftShape, rightShape):
    global tests

    maxDims = max(len(leftShape), len(rightShape))

    # op0 = ExtendShape(leftShape, maxDims)
    # op1 = ExtendShape(rightShape, maxDims)

    # broadCastedShape = BroadCastShape(op0, op1)

    # Let onnx infer shape specifics
    outputShape = [None] * maxDims

    testIndex = len(tests)

    test = Test()
    test.shapes = [leftShape, rightShape]

    leftTensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 0), TensorProto.FLOAT, leftShape
    )
    rightTensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 1), TensorProto.FLOAT, rightShape
    )

    test.tensors = [leftTensor, rightTensor]
    test.outputTensor = make_tensor_value_info(
        GetOutputTrueName(testIndex), TensorProto.FLOAT, outputShape
    )
    test.node = make_node(
        op,
        [GetInputTrueName(testIndex, 0), GetInputTrueName(testIndex, 1)],
        [GetOutputTrueName(testIndex)],
    )

    leftRandomArray = np.random.randn(*leftShape).astype(np.float32)
    rightRandomArray = np.random.randn(*rightShape).astype(np.float32)
    test.randomArrays = [leftRandomArray, rightRandomArray]

    tests.append(test)


def CreateUnaryOpTest(op, shape):
    global tests
    testIndex = len(tests)

    test = Test()
    test.shapes = [shape]

    tensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 0), TensorProto.FLOAT, shape
    )

    test.tensors = [tensor]
    test.outputTensor = make_tensor_value_info(
        GetOutputTrueName(testIndex), TensorProto.FLOAT, shape
    )
    test.node = make_node(
        op, [GetInputTrueName(testIndex, 0)], [GetOutputTrueName(testIndex)]
    )

    randomArray = np.random.randn(*shape).astype(np.float32)
    test.randomArrays = [randomArray]

    tests.append(test)


def CreateMaxPool(shape, kernel, strides, auto_pad="NOTSET", pads=None):
    global tests
    testIndex = len(tests)

    test = Test()
    test.shapes = [shape]

    tensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 0), TensorProto.FLOAT, shape
    )

    # Let onnx infer shape specifics
    outputShape = [None] * len(shape)

    if auto_pad == "NOTSET":
        if pads == None:
            pads = [0, 0, 0, 0]

    if pads:
        assert auto_pad == "NOTSET"

    test.tensors = [tensor]
    test.outputTensor = make_tensor_value_info(
        GetOutputTrueName(testIndex), TensorProto.FLOAT, outputShape
    )
    test.node = make_node(
        "MaxPool",
        [GetInputTrueName(testIndex, 0)],
        [GetOutputTrueName(testIndex)],
        kernel_shape=kernel,
        strides=strides,
        auto_pad=auto_pad,
        pads=pads,
    )

    randomArray = np.random.randn(*shape).astype(np.float32)
    test.randomArrays = [randomArray]

    tests.append(test)


def CreateAveragePool(shape, kernel, strides, auto_pad="NOTSET", pads=None):
    global tests
    testIndex = len(tests)

    test = Test()
    test.shapes = [shape]

    tensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 0), TensorProto.FLOAT, shape
    )

    # Let onnx infer shape specifics
    outputShape = [None] * len(shape)

    if auto_pad == "NOTSET":
        if pads == None:
            pads = [0, 0, 0, 0]

    if pads:
        assert auto_pad == "NOTSET"

    test.tensors = [tensor]
    test.outputTensor = make_tensor_value_info(
        GetOutputTrueName(testIndex), TensorProto.FLOAT, outputShape
    )
    test.node = make_node(
        "AveragePool",
        [GetInputTrueName(testIndex, 0)],
        [GetOutputTrueName(testIndex)],
        kernel_shape=kernel,
        strides=strides,
        auto_pad=auto_pad,
        pads=pads,
    )

    randomArray = np.random.randn(*shape).astype(np.float32)
    test.randomArrays = [randomArray]

    tests.append(test)


def CreateConvolution(
    shape,
    features,
    kernel,
    strides,
    dilations,
    bias: bool = False,
    auto_pad="NOTSET",
    pads=None,
):
    global tests
    testIndex = len(tests)

    test = Test()
    test.shapes = [shape]

    inputTensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 0), TensorProto.FLOAT, shape
    )

    kernelShape = [features, shape[1], kernel[0], kernel[1]]
    kernelTensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 1), TensorProto.FLOAT, kernelShape
    )

    biasTensor = None
    if bias:
        biasTensor = make_tensor_value_info(
            GetInputTrueName(testIndex, 2), TensorProto.FLOAT, [features]
        )

    # Let onnx infer shape specifics
    outputShape = [None] * len(shape)

    if auto_pad == "NOTSET":
        if pads == None:
            pads = [0, 0, 0, 0]

    if pads:
        assert auto_pad == "NOTSET"

    test.tensors = [inputTensor, kernelTensor]
    if bias:
        test.tensors.append(biasTensor)

    test.outputTensor = make_tensor_value_info(
        GetOutputTrueName(testIndex), TensorProto.FLOAT, outputShape
    )
    inputs = [GetInputTrueName(testIndex, 0), GetInputTrueName(testIndex, 1)]
    if bias:
        inputs.append(GetInputTrueName(testIndex, 2))

    test.node = make_node(
        "Conv",
        inputs,
        [GetOutputTrueName(testIndex)],
        auto_pad=auto_pad,
        dilations=dilations,
        kernel_shape=kernel,
        pads=pads,
        strides=strides,
    )

    randomArray0 = np.random.randn(*shape).astype(np.float32)
    randomArray1 = np.random.randn(*[features, shape[1], kernel[0], kernel[1]]).astype(
        np.float32
    )

    test.randomArrays = [randomArray0, randomArray1]

    if bias:
        randomBias = np.random.randn(*[features]).astype(np.float32)
        test.randomArrays.append(randomBias)

    tests.append(test)


def CreateReshape(shapeIn, shapeOut):
    global tests
    testIndex = len(tests)

    test = Test()

    val = np.array(shapeOut, dtype=np.int64)
    A = numpy_helper.from_array(val, name=GetInitializerTrueName(testIndex))

    test.shapes = [shapeIn]

    tensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 0), TensorProto.FLOAT, shapeIn
    )

    test.tensors = [tensor]
    test.outputTensor = make_tensor_value_info(
        GetOutputTrueName(testIndex), TensorProto.FLOAT, shapeOut
    )
    test.node = make_node(
        "Reshape",
        [GetInputTrueName(testIndex, 0), GetInitializerTrueName(testIndex)],
        [GetOutputTrueName(testIndex)],
    )

    randomArray = np.random.randn(*shapeIn).astype(np.float32)
    test.randomArrays = [randomArray]
    test.initializerArrays = [A]

    tests.append(test)


def CreateSoftmax(shape, axis=-1):
    global tests
    testIndex = len(tests)

    test = Test()

    test.shapes = [shape]
    tensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 0), TensorProto.FLOAT, shape
    )
    test.tensors = [tensor]

    shapeOut = [None] * len(shape)
    test.outputTensor = make_tensor_value_info(
        GetOutputTrueName(testIndex), TensorProto.FLOAT, shapeOut
    )
    test.node = make_node(
        "Softmax",
        [GetInputTrueName(testIndex, 0)],
        [GetOutputTrueName(testIndex)],
        axis=axis,
    )

    randomArray = np.random.randn(*shape).astype(np.float32)
    test.randomArrays = [randomArray]

    tests.append(test)


testAdd = True
testRelu = False
testReshape = False
testMatMul = False
testMaxPool = False
testConv = False
testAveragePool = False
testSoftmax = False

if __name__ == "__main__":
    if testSoftmax:
        # Softmax axis come in pairs.
        # If the dim is N, then the pairs are X and X - N.
        # For dim=2, pairs are 0,-2 and 1,-1
        # For dim=3, pairs are 0,-3 , 1,-2 and 2,-1
        # Softmax sums everything to the "right" of (and including) the axis used.
        # For the 2D example, 0 is everything
        # While 1 is everything to the right of the y dim (which means that we iterate the y dim).

        # For the 3D example, 0 is everything
        #                     1 is everything right of Z (iterate Z).
        #                     2 is everything right of Y (iterate Z and Y).

        w = 5
        z = 4
        y = 3
        x = 2

        CreateSoftmax([1], 0)
        CreateSoftmax([2], 0)
        CreateSoftmax([10], 0)

        CreateSoftmax([y, x], -2)  # A
        CreateSoftmax([y, x], -1)  # B
        CreateSoftmax([y, x], 0)  # A
        CreateSoftmax([y, x], 1)  # B

        CreateSoftmax([z, y, x], -3)  # A
        CreateSoftmax([z, y, x], -2)  # B
        CreateSoftmax([z, y, x], -1)  # C
        CreateSoftmax([z, y, x], 0)  # A
        CreateSoftmax([z, y, x], 1)  # B
        CreateSoftmax([z, y, x], 2)  # C

        CreateSoftmax([w, z, y, x], -4)  # A
        CreateSoftmax([w, z, y, x], -3)  # B
        CreateSoftmax([w, z, y, x], -2)  # C
        CreateSoftmax([w, z, y, x], -1)  # D
        CreateSoftmax([w, z, y, x], 0)  # A
        CreateSoftmax([w, z, y, x], 1)  # B
        CreateSoftmax([w, z, y, x], 2)  # C
        CreateSoftmax([w, z, y, x], 3)  # D

    if testAdd:
        # Simplest tests, no broadcast or abusing dimensions
        CreateBinaryOpTest("Add", [1], [1])
        CreateBinaryOpTest("Add", [4], [4])
        CreateBinaryOpTest("Add", [2, 4], [2, 4])
        CreateBinaryOpTest("Add", [2, 4, 6], [2, 4, 6])
        CreateBinaryOpTest("Add", [2, 4, 6, 8], [2, 4, 6, 8])

        # Broadcasting
        CreateBinaryOpTest("Add", [2, 3, 4, 5], [1])
        CreateBinaryOpTest("Add", [2, 3, 4, 5], [5])
        CreateBinaryOpTest("Add", [4, 5], [2, 3, 4, 5])
        CreateBinaryOpTest("Add", [1, 4, 5], [2, 3, 1, 1])
        CreateBinaryOpTest("Add", [3, 4, 5], [2, 1, 1, 1])

    if testRelu:
        CreateUnaryOpTest("Relu", [1])
        CreateUnaryOpTest("Relu", [4])
        CreateUnaryOpTest("Relu", [2, 4])
        CreateUnaryOpTest("Relu", [2, 4, 6])
        CreateUnaryOpTest("Relu", [2, 4, 6, 8])

    if testReshape:
        CreateReshape([4, 2], [8])
        CreateReshape([4, 2], [2, 4])
        CreateReshape([1, 8], [8])
        CreateReshape([2, 3, 4], [24])
        CreateReshape([24], [2, 3, 4])
        CreateReshape([24], [4, 3, 2])

    if testMatMul:
        CreateBinaryOpTest("MatMul", [1, 1], [1, 1])
        CreateBinaryOpTest("MatMul", [1, 2], [2, 1])
        CreateBinaryOpTest("MatMul", [2, 1], [1, 2])
        CreateBinaryOpTest("MatMul", [2, 2], [2, 2])
        CreateBinaryOpTest("MatMul", [2, 3], [3, 2])
        CreateBinaryOpTest("MatMul", [3, 2], [2, 3])
        CreateBinaryOpTest("MatMul", [2, 4], [4, 8])
        CreateBinaryOpTest("MatMul", [8, 4], [4, 2])
        CreateBinaryOpTest("MatMul", [10, 11], [11, 20])

    if testMaxPool:
        # All padding posibilities, mostly to test the window generation
        # No padding                                           T  L  B  R
        CreateMaxPool([1, 1, 4, 4], [2, 2], [2, 2], "NOTSET", [0, 0, 0, 0])
        CreateMaxPool([1, 1, 3, 4], [2, 2], [2, 2], "NOTSET", [1, 0, 0, 0])
        CreateMaxPool([1, 1, 4, 3], [2, 2], [2, 2], "NOTSET", [0, 1, 0, 0])
        CreateMaxPool([1, 1, 3, 4], [2, 2], [2, 2], "NOTSET", [0, 0, 1, 0])
        CreateMaxPool([1, 1, 4, 3], [2, 2], [2, 2], "NOTSET", [0, 0, 0, 1])
        CreateMaxPool([1, 1, 3, 3], [2, 2], [2, 2], "NOTSET", [1, 1, 0, 0])
        CreateMaxPool([1, 1, 2, 4], [2, 2], [2, 2], "NOTSET", [1, 0, 1, 0])
        CreateMaxPool([1, 1, 3, 3], [2, 2], [2, 2], "NOTSET", [1, 0, 0, 1])
        CreateMaxPool([1, 1, 3, 3], [2, 2], [2, 2], "NOTSET", [0, 1, 1, 0])
        CreateMaxPool([1, 1, 4, 2], [2, 2], [2, 2], "NOTSET", [0, 1, 0, 1])
        CreateMaxPool([1, 1, 3, 3], [2, 2], [2, 2], "NOTSET", [0, 0, 1, 1])
        CreateMaxPool([1, 1, 2, 3], [2, 2], [2, 2], "NOTSET", [1, 1, 1, 0])
        CreateMaxPool([1, 1, 3, 2], [2, 2], [2, 2], "NOTSET", [1, 1, 0, 1])
        CreateMaxPool([1, 1, 2, 3], [2, 2], [2, 2], "NOTSET", [1, 0, 1, 1])
        CreateMaxPool([1, 1, 3, 2], [2, 2], [2, 2], "NOTSET", [0, 1, 1, 1])
        CreateMaxPool([1, 1, 2, 2], [2, 2], [2, 2], "NOTSET", [1, 1, 1, 1])
        CreateMaxPool([1, 1, 1, 1], [3, 3], [3, 3], "NOTSET", [1, 1, 1, 1])
        CreateMaxPool([1, 1, 10, 10], [2, 2], [2, 2], "NOTSET", [1, 1, 1, 1])

        # Test different kernels, strides, no padding
        CreateMaxPool([1, 3, 8, 8], [2, 2], [2, 2])
        CreateMaxPool([1, 3, 9, 9], [3, 3], [3, 3])
        CreateMaxPool([1, 3, 9, 8], [3, 2], [3, 2])
        CreateMaxPool([1, 3, 8, 9], [2, 3], [2, 3])

        # Different auto pad, minimal size
        CreateMaxPool([1, 3, 1, 1], [2, 2], [2, 2], "SAME_UPPER")
        CreateMaxPool([1, 3, 3, 3], [2, 2], [2, 2], "SAME_UPPER")
        CreateMaxPool([1, 3, 5, 5], [2, 2], [2, 2], "SAME_UPPER")

        CreateMaxPool([1, 3, 1, 1], [2, 2], [2, 2], "SAME_LOWER")
        CreateMaxPool([1, 3, 3, 3], [2, 2], [2, 2], "SAME_LOWER")
        CreateMaxPool([1, 3, 5, 5], [2, 2], [2, 2], "SAME_LOWER")

        # Different auto pad, larger size
        CreateMaxPool([1, 3, 8, 8], [3, 2], [2, 3], "SAME_UPPER")
        CreateMaxPool([1, 3, 8, 8], [2, 3], [3, 2], "SAME_UPPER")
        CreateMaxPool([1, 3, 8, 8], [3, 3], [2, 2], "SAME_UPPER")
        CreateMaxPool([1, 3, 8, 8], [2, 2], [3, 3], "SAME_UPPER")
        CreateMaxPool([1, 3, 8, 8], [3, 2], [2, 3], "SAME_LOWER")
        CreateMaxPool([1, 3, 8, 8], [2, 3], [3, 2], "SAME_LOWER")
        CreateMaxPool([1, 3, 8, 8], [3, 3], [2, 2], "SAME_LOWER")
        CreateMaxPool([1, 3, 8, 8], [2, 2], [3, 3], "SAME_LOWER")
        CreateMaxPool([1, 3, 8, 8], [3, 2], [2, 3], "VALID")
        CreateMaxPool([1, 3, 8, 8], [2, 3], [3, 2], "VALID")
        CreateMaxPool([1, 3, 8, 8], [3, 3], [2, 2], "VALID")
        CreateMaxPool([1, 3, 8, 8], [2, 2], [3, 3], "VALID")

        # Larger example and larger kernel
        CreateMaxPool([1, 3, 5, 5], [20, 20], [20, 20], "SAME_UPPER")
        CreateMaxPool([1, 3, 5, 5], [30, 20], [20, 30], "SAME_UPPER")
        CreateMaxPool([1, 3, 5, 5], [20, 30], [30, 20], "SAME_UPPER")
        CreateMaxPool([1, 3, 5, 5], [30, 30], [20, 20], "SAME_UPPER")
        CreateMaxPool([1, 3, 5, 5], [20, 20], [30, 30], "SAME_UPPER")

        CreateMaxPool([1, 3, 5, 5], [20, 20], [20, 20], "SAME_LOWER")
        CreateMaxPool([1, 3, 5, 5], [30, 20], [20, 30], "SAME_LOWER")
        CreateMaxPool([1, 3, 5, 5], [20, 30], [30, 20], "SAME_LOWER")
        CreateMaxPool([1, 3, 5, 5], [30, 30], [20, 20], "SAME_LOWER")
        CreateMaxPool([1, 3, 5, 5], [20, 20], [30, 30], "SAME_LOWER")

        # Common example
        CreateMaxPool([1, 3, 32, 32], [2, 2], [2, 2], "VALID")

        # 3 D
        # CreateMaxPool([1, 3, 8, 8, 8], [2, 2, 2], [2, 2, 2])

        # 4 D - Not supported by runtime so cannot generate test

    if testAveragePool:
        # All padding posibilities, mostly to test the window generation
        # No padding                                               T  L  B  R
        CreateAveragePool([1, 1, 4, 4], [2, 2], [2, 2], "NOTSET", [0, 0, 0, 0])
        CreateAveragePool([1, 1, 3, 4], [2, 2], [2, 2], "NOTSET", [1, 0, 0, 0])
        CreateAveragePool([1, 1, 4, 3], [2, 2], [2, 2], "NOTSET", [0, 1, 0, 0])
        CreateAveragePool([1, 1, 3, 4], [2, 2], [2, 2], "NOTSET", [0, 0, 1, 0])
        CreateAveragePool([1, 1, 4, 3], [2, 2], [2, 2], "NOTSET", [0, 0, 0, 1])
        CreateAveragePool([1, 1, 3, 3], [2, 2], [2, 2], "NOTSET", [1, 1, 0, 0])
        CreateAveragePool([1, 1, 2, 4], [2, 2], [2, 2], "NOTSET", [1, 0, 1, 0])
        CreateAveragePool([1, 1, 3, 3], [2, 2], [2, 2], "NOTSET", [1, 0, 0, 1])
        CreateAveragePool([1, 1, 3, 3], [2, 2], [2, 2], "NOTSET", [0, 1, 1, 0])
        CreateAveragePool([1, 1, 4, 2], [2, 2], [2, 2], "NOTSET", [0, 1, 0, 1])
        CreateAveragePool([1, 1, 3, 3], [2, 2], [2, 2], "NOTSET", [0, 0, 1, 1])
        CreateAveragePool([1, 1, 2, 3], [2, 2], [2, 2], "NOTSET", [1, 1, 1, 0])
        CreateAveragePool([1, 1, 3, 2], [2, 2], [2, 2], "NOTSET", [1, 1, 0, 1])
        CreateAveragePool([1, 1, 2, 3], [2, 2], [2, 2], "NOTSET", [1, 0, 1, 1])
        CreateAveragePool([1, 1, 3, 2], [2, 2], [2, 2], "NOTSET", [0, 1, 1, 1])
        CreateAveragePool([1, 1, 2, 2], [2, 2], [2, 2], "NOTSET", [1, 1, 1, 1])
        CreateAveragePool([1, 1, 1, 1], [3, 3], [3, 3], "NOTSET", [1, 1, 1, 1])
        CreateAveragePool([1, 1, 10, 10], [2, 2], [2, 2], "NOTSET", [1, 1, 1, 1])

        # Test different kernels, strides, no padding
        CreateAveragePool([1, 3, 8, 8], [2, 2], [2, 2])
        CreateAveragePool([1, 3, 9, 9], [3, 3], [3, 3])
        CreateAveragePool([1, 3, 9, 8], [3, 2], [3, 2])
        CreateAveragePool([1, 3, 8, 9], [2, 3], [2, 3])

        # Simple padding example, kernel matches stride
        CreateAveragePool([1, 3, 1, 1], [2, 2], [2, 2], "SAME_UPPER")
        CreateAveragePool([1, 3, 3, 3], [2, 2], [2, 2], "SAME_UPPER")
        CreateAveragePool([1, 3, 5, 5], [2, 2], [2, 2], "SAME_UPPER")

        CreateAveragePool([1, 3, 1, 1], [2, 2], [2, 2], "SAME_LOWER")
        CreateAveragePool([1, 3, 3, 3], [2, 2], [2, 2], "SAME_LOWER")
        CreateAveragePool([1, 3, 5, 5], [2, 2], [2, 2], "SAME_LOWER")

        # Larger size
        CreateAveragePool([1, 3, 8, 8], [3, 2], [2, 3], "SAME_UPPER")
        CreateAveragePool([1, 3, 8, 8], [2, 3], [3, 2], "SAME_UPPER")
        CreateAveragePool([1, 3, 8, 8], [3, 3], [2, 2], "SAME_UPPER")
        CreateAveragePool([1, 3, 8, 8], [2, 2], [3, 3], "SAME_UPPER")
        CreateAveragePool([1, 3, 8, 8], [3, 2], [2, 3], "SAME_LOWER")
        CreateAveragePool([1, 3, 8, 8], [2, 3], [3, 2], "SAME_LOWER")
        CreateAveragePool([1, 3, 8, 8], [3, 3], [2, 2], "SAME_LOWER")
        CreateAveragePool([1, 3, 8, 8], [2, 2], [3, 3], "SAME_LOWER")
        CreateAveragePool([1, 3, 8, 8], [3, 2], [2, 3], "VALID")
        CreateAveragePool([1, 3, 8, 8], [2, 3], [3, 2], "VALID")
        CreateAveragePool([1, 3, 8, 8], [3, 3], [2, 2], "VALID")
        CreateAveragePool([1, 3, 8, 8], [2, 2], [3, 3], "VALID")

        CreateAveragePool([1, 3, 5, 5], [20, 20], [20, 20], "SAME_UPPER")
        CreateAveragePool([1, 3, 5, 5], [30, 20], [20, 30], "SAME_UPPER")
        CreateAveragePool([1, 3, 5, 5], [20, 30], [30, 20], "SAME_UPPER")
        CreateAveragePool([1, 3, 5, 5], [30, 30], [20, 20], "SAME_UPPER")
        CreateAveragePool([1, 3, 5, 5], [20, 20], [30, 30], "SAME_UPPER")

        CreateAveragePool([1, 3, 5, 5], [20, 20], [20, 20], "SAME_LOWER")
        CreateAveragePool([1, 3, 5, 5], [30, 20], [20, 30], "SAME_LOWER")
        CreateAveragePool([1, 3, 5, 5], [20, 30], [30, 20], "SAME_LOWER")
        CreateAveragePool([1, 3, 5, 5], [30, 30], [20, 20], "SAME_LOWER")
        CreateAveragePool([1, 3, 5, 5], [20, 20], [30, 30], "SAME_LOWER")

        # Common example
        CreateAveragePool([1, 3, 32, 32], [2, 2], [2, 2], "VALID")

        # 3 D
        # CreateAveragePool([1, 3, 8, 8, 8], [2, 2, 2], [2, 2, 2])

        # 4 D - Not supported by runtime, so cannot generate the test

    # Convolution
    if testConv:
        # All padding posibilities, mostly to test the window generation
        # Input shape, features, kernel, stride, dilations, bias
        n = 1
        c = 3
        f = 16

        k = [3, 3]
        s = [3, 3]
        d = [1, 1]
        b = False
        p = "NOTSET"

        if False:
            #                                                  T  L  B  R
            CreateConvolution([n, c, 6, 6], f, k, s, d, b, p, [0, 0, 0, 0])
            CreateConvolution([n, c, 5, 6], f, k, s, d, b, p, [1, 0, 0, 0])
            CreateConvolution([n, c, 6, 5], f, k, s, d, b, p, [0, 1, 0, 0])
            CreateConvolution([n, c, 5, 6], f, k, s, d, b, p, [0, 0, 1, 0])
            CreateConvolution([n, c, 6, 5], f, k, s, d, b, p, [0, 0, 0, 1])
            CreateConvolution([n, c, 5, 5], f, k, s, d, b, p, [1, 1, 0, 0])
            CreateConvolution([n, c, 4, 6], f, k, s, d, b, p, [1, 0, 1, 0])
            CreateConvolution([n, c, 5, 5], f, k, s, d, b, p, [1, 0, 0, 1])
            CreateConvolution([n, c, 5, 5], f, k, s, d, b, p, [0, 1, 1, 0])
            CreateConvolution([n, c, 6, 4], f, k, s, d, b, p, [0, 1, 0, 1])
            CreateConvolution([n, c, 5, 5], f, k, s, d, b, p, [0, 0, 1, 1])
            CreateConvolution([n, c, 4, 5], f, k, s, d, b, p, [1, 1, 1, 0])
            CreateConvolution([n, c, 5, 4], f, k, s, d, b, p, [1, 1, 0, 1])
            CreateConvolution([n, c, 4, 5], f, k, s, d, b, p, [1, 0, 1, 1])
            CreateConvolution([n, c, 5, 4], f, k, s, d, b, p, [0, 1, 1, 1])
            CreateConvolution([n, c, 4, 4], f, k, s, d, b, p, [1, 1, 1, 1])
            CreateConvolution([n, c, 1, 1], f, k, s, d, b, p, [1, 1, 1, 1])
            CreateConvolution([n, c, 10, 10], f, k, s, d, b, p, [1, 1, 1, 1])

            # No padding
            # Different: Input shape, features, kernel, stride, dilations, bias
            CreateConvolution([1, 1, 3, 3], 1, [3, 3], [3, 3], d)
            CreateConvolution([1, 2, 3, 3], 1, [3, 3], [3, 3], d)
            CreateConvolution([1, 1, 3, 3], 2, [3, 3], [3, 3], d)
            CreateConvolution([1, 2, 3, 3], 2, [3, 3], [3, 3], d)

            # Same but in a 2x2 square
            CreateConvolution([1, 1, 6, 6], 1, [3, 3], [3, 3], d)
            CreateConvolution([1, 2, 6, 6], 1, [3, 3], [3, 3], d)
            CreateConvolution([1, 1, 6, 6], 2, [3, 3], [3, 3], d)
            CreateConvolution([1, 2, 6, 6], 2, [3, 3], [3, 3], d)

            # Same but for a 5x5 kernel
            CreateConvolution([1, 1, 5, 5], 1, [5, 5], [5, 5], d)
            CreateConvolution([1, 2, 5, 5], 1, [5, 5], [5, 5], d)
            CreateConvolution([1, 1, 5, 5], 2, [5, 5], [5, 5], d)
            CreateConvolution([1, 2, 5, 5], 2, [5, 5], [5, 5], d)

            # Same but for a 2x2 kernel with stride of 1x1 (result is 3x3)
            CreateConvolution([1, 1, 4, 4], 1, [2, 2], [1, 1], d)
            CreateConvolution([1, 2, 4, 4], 1, [2, 2], [1, 1], d)
            CreateConvolution([1, 1, 4, 4], 2, [2, 2], [1, 1], d)
            CreateConvolution([1, 2, 4, 4], 2, [2, 2], [1, 1], d)

            # Different sized kernels
            CreateConvolution([1, 1, 2, 3], 1, [2, 3], [2, 3], d)
            CreateConvolution([1, 1, 3, 2], 1, [3, 2], [3, 2], d)
            CreateConvolution([1, 1, 4, 9], 1, [2, 3], [2, 3], d)
            CreateConvolution([1, 1, 9, 4], 1, [3, 2], [3, 2], d)

            # Bigger more realistic examples
            CreateConvolution([1, 3, 16, 16], 16, [2, 2], [2, 2], d)

        CreateConvolution([1, 1, 1, 1], 2, [5, 5], [5, 5], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 1, 1], 2, [5, 5], [1, 1], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 1, 1], 1, [5, 5], [1, 1], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 3, 3], 1, [5, 5], [1, 1], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 5, 5], 1, [5, 5], [1, 1], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 8, 8], 2, [5, 5], [1, 1], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 10, 10], 2, [5, 5], [1, 1], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 15, 15], 2, [5, 5], [1, 1], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 20, 20], 2, [5, 5], [1, 1], d, False, "SAME_UPPER")
        CreateConvolution([1, 1, 28, 28], 2, [5, 5], [1, 1], d, False, "SAME_UPPER")

        # Adding bias
        CreateConvolution([1, 1, 3, 3], 1, [3, 3], [3, 3], d, True)
        CreateConvolution([1, 2, 3, 3], 1, [3, 3], [3, 3], d, True)
        CreateConvolution([1, 1, 3, 3], 2, [3, 3], [3, 3], d, True)
        CreateConvolution([1, 2, 3, 3], 2, [3, 3], [3, 3], d, True)
        CreateConvolution([1, 1, 2, 3], 1, [2, 3], [2, 3], d, True)
        CreateConvolution([1, 1, 3, 2], 1, [3, 2], [3, 2], d, True)
        CreateConvolution([1, 1, 4, 9], 1, [2, 3], [2, 3], d, True)
        CreateConvolution([1, 1, 9, 4], 1, [3, 2], [3, 2], d, True)

    allInputNodesAndValuesInOrder = []
    for x in tests:
        for tensor, randomArray in zip(x.tensors, x.randomArrays):
            allInputNodesAndValuesInOrder.append([tensor, randomArray])

    allNodes = [x.node for x in tests]
    allInputNodes = [x[0] for x in allInputNodesAndValuesInOrder]
    allOutputNodes = [x.outputTensor for x in tests]

    allInitializers = []
    for test in tests:
        if test.initializerArrays:
            for x in test.initializerArrays:
                allInitializers.append(x)

    graph = make_graph(
        allNodes, "simpleTest", allInputNodes, allOutputNodes, allInitializers
    )

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 7)])
    check_model(onnx_model)
    onnx_model = version_converter.convert_version(onnx_model, 7)
    check_model(onnx_model)
    shaped = onnx.shape_inference.infer_shapes(onnx_model)
    check_model(shaped)

    outputPath = sys.argv[1]
    try:
        shutil.rmtree(outputPath)
    except:
        pass

    try:
        os.makedirs(os.path.join(outputPath, "test_data_set_0"))
    except FileNotFoundError:
        print(f"Error creating path for output: {outputPath}")
        sys.exit(0)
    except FileExistsError:
        pass  # Not a problem if folder already exists.

    for i, nodeAndValue in enumerate(allInputNodesAndValuesInOrder):
        value = nodeAndValue[1]
        with open(os.path.join(outputPath, f"test_data_set_0/input_{i}.pb"), "wb") as f:
            asTensor = numpy_helper.from_array(value)
            f.write(asTensor.SerializeToString())

    sess = ort.InferenceSession(shaped.SerializeToString())

    modelInputs = {}
    for i, test in enumerate(tests):
        for j, randomArray in enumerate(test.randomArrays):
            modelInputs[GetInputTrueName(i, j)] = randomArray

    modelOutput = sess.run(None, modelInputs)

    for i in range(len(tests)):
        with open(
            os.path.join(outputPath, f"test_data_set_0/output_{i}.pb"), "wb"
        ) as f:
            asTensor = numpy_helper.from_array(modelOutput[i])
            f.write(asTensor.SerializeToString())

    save_onnx_model(shaped, os.path.join(outputPath, "model.onnx"))
