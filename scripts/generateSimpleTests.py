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

    op0 = ExtendShape(leftShape, maxDims)
    op1 = ExtendShape(rightShape, maxDims)

    broadCastedShape = BroadCastShape(op0, op1)

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
        GetOutputTrueName(testIndex), TensorProto.FLOAT, broadCastedShape
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

    if(auto_pad == "NOTSET"):
        assert(pads)

    if(pads):
        assert(auto_pad == "NOTSET")

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
        pads=pads
    )

    randomArray = np.random.randn(*shape).astype(np.float32)
    test.randomArrays = [randomArray]

    tests.append(test)

def CreateConvolution(shape, features, kernel, strides, dilations, bias: bool, auto_pad="NOTSET", pads=None):
    global tests
    testIndex = len(tests)

    test = Test()
    test.shapes = [shape]

    inputTensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 0), TensorProto.FLOAT, shape
    )

    kernelShape = [features,shape[1],kernel[0],kernel[1]]
    kernelTensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 1), TensorProto.FLOAT, kernelShape
    )

    biasTensor = None
    if(bias):
        biasTensor = make_tensor_value_info(
        GetInputTrueName(testIndex, 2), TensorProto.FLOAT, [features]
    )

    # Let onnx infer shape specifics
    outputShape = [None] * len(shape)

    if(auto_pad == "NOTSET"):
        assert(pads)

    if(pads):
        assert(auto_pad == "NOTSET")

    test.tensors = [inputTensor,kernelTensor]
    if(bias):
        test.tensors.append(biasTensor)

    test.outputTensor = make_tensor_value_info(
        GetOutputTrueName(testIndex), TensorProto.FLOAT, outputShape
    )
    inputs = [GetInputTrueName(testIndex, 0),GetInputTrueName(testIndex, 1)]
    if(bias):
        inputs.append(GetInputTrueName(testIndex, 2))

    test.node = make_node(
        "Conv",
        inputs,
        [GetOutputTrueName(testIndex)],
        auto_pad=auto_pad,
        dilations=dilations,
        kernel_shape=kernel,
        pads=pads,
        strides=strides
    )

    randomArray0 = np.random.randn(*shape).astype(np.float32)
    randomArray1 = np.random.randn(*[features,shape[1],kernel[0],kernel[1]]).astype(np.float32)

    test.randomArrays = [randomArray0,randomArray1]

    if(bias):
        randomBias =  np.random.randn(*[features]).astype(np.float32)
        test.randomArrays.append(randomBias)

    tests.append(test)

def CreateReshapeTest(shapeIn, shapeOut):
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


if __name__ == "__main__":

    # Add tests
    if False:
        if True:
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

    # Relu 
    if False:
        CreateUnaryOpTest("Relu", [1])
        CreateUnaryOpTest("Relu", [4])
        CreateUnaryOpTest("Relu", [2, 4])
        CreateUnaryOpTest("Relu", [2, 4, 6])
        CreateUnaryOpTest("Relu", [2, 4, 6, 8])

    # Reshape
    if False:
        CreateReshapeTest([4, 2], [8])
        CreateReshapeTest([4, 2], [2, 4])
        CreateReshapeTest([1, 8], [8])
        CreateReshapeTest([2, 3, 4], [24])
        CreateReshapeTest([24], [2, 3, 4])
        CreateReshapeTest([24], [4, 3, 2])

    # MaxPool
    if False:
        if True:
            # Test different kernels, strides, no padding
            CreateMaxPool([1, 3, 8, 8], [2, 2], [2, 2],"NOTSET",[0,0,0,0])
            CreateMaxPool([1, 3, 9, 9], [3, 3], [3, 3],"NOTSET",[0,0,0,0])
            CreateMaxPool([1, 3, 9, 8], [3, 2], [3, 2],"NOTSET",[0,0,0,0])
            CreateMaxPool([1, 3, 8, 9], [2, 3], [2, 3],"NOTSET",[0,0,0,0])

            # Simple padding example, kernel matches stride
        if True:
            # When in notset, we ignore values. (A [5,5] image with a [2,2] stride generates a [2,2] image)
            # The exception appears to be a [1,1] image, where we produce a [1,1] output
            CreateMaxPool([1, 3, 1, 1], [2, 2], [2, 2],"NOTSET",[0,0,1,1])
            CreateMaxPool([1, 3, 3, 3], [2, 2], [2, 2],"NOTSET",[0,0,1,1])
            CreateMaxPool([1, 3, 5, 5], [2, 2], [2, 2],"NOTSET",[0,0,1,1])

        if True:
            CreateMaxPool([1, 3, 1, 1], [2, 2], [2, 2],"SAME_UPPER")
            CreateMaxPool([1, 3, 3, 3], [2, 2], [2, 2],"SAME_UPPER")
            CreateMaxPool([1, 3, 5, 5], [2, 2], [2, 2],"SAME_UPPER")

        if True:
            CreateMaxPool([1, 3, 1, 1], [2, 2], [2, 2],"SAME_LOWER")
            CreateMaxPool([1, 3, 3, 3], [2, 2], [2, 2],"SAME_LOWER")
            CreateMaxPool([1, 3, 5, 5], [2, 2], [2, 2],"SAME_LOWER")

        if True:
            CreateMaxPool([1, 3, 1, 1], [2, 2], [2, 2],"NOTSET",[0,0,1,1]) # Should be the same as SAME_UPPER
            CreateMaxPool([1, 3, 1, 1], [2, 2], [2, 2],"NOTSET",[1,1,0,0]) # Should be the same as SAME_LOWER

        if True:
            CreateMaxPool([1, 3, 8, 8], [3, 2], [2, 3],"SAME_UPPER")
            CreateMaxPool([1, 3, 8, 8], [2, 3], [3, 2],"SAME_UPPER")
            CreateMaxPool([1, 3, 8, 8], [3, 3], [2, 2],"SAME_UPPER")
            CreateMaxPool([1, 3, 8, 8], [2, 2], [3, 3],"SAME_UPPER")
            CreateMaxPool([1, 3, 8, 8], [3, 2], [2, 3],"SAME_LOWER")
            CreateMaxPool([1, 3, 8, 8], [2, 3], [3, 2],"SAME_LOWER")
            CreateMaxPool([1, 3, 8, 8], [3, 3], [2, 2],"SAME_LOWER")
            CreateMaxPool([1, 3, 8, 8], [2, 2], [3, 3],"SAME_LOWER")
            CreateMaxPool([1, 3, 8, 8], [3, 2], [2, 3],"VALID")
            CreateMaxPool([1, 3, 8, 8], [2, 3], [3, 2],"VALID")
            CreateMaxPool([1, 3, 8, 8], [3, 3], [2, 2],"VALID")
            CreateMaxPool([1, 3, 8, 8], [2, 2], [3, 3],"VALID")

        if True:
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

            CreateMaxPool([1, 3, 5, 5], [20, 20], [20, 20], "VALID")

            # 3 D
            # CreateMaxPool([1, 3, 8, 8, 8], [2, 2, 2], [2, 2, 2])

            # 4 D - Not supported by runtime, so cannot generate the test
            # CreateMaxPool([1, 3, 8, 8, 8, 8], [2, 2, 2, 2], [2, 2, 2, 2])

    # Convolution
    if True:
        # No padding
        # Different: Input shape, features, kernel, stride, dilations, bias

        if True:
            # First test, changing input channels and features
            if True:
                CreateConvolution([1,1,3,3],1,[3,3],[3,3],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,2,3,3],1,[3,3],[3,3],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,1,3,3],2,[3,3],[3,3],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,2,3,3],2,[3,3],[3,3],[1,1],False,"NOTSET",[0,0,0,0])

            # Same but in a 2x2 square
            if True:
                CreateConvolution([1,1,6,6],1,[3,3],[3,3],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,2,6,6],1,[3,3],[3,3],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,1,6,6],2,[3,3],[3,3],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,2,6,6],2,[3,3],[3,3],[1,1],False,"NOTSET",[0,0,0,0])

            # Same but for a 5x5 kernel 
            if True:
                CreateConvolution([1,1,5,5],1,[5,5],[5,5],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,2,5,5],1,[5,5],[5,5],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,1,5,5],2,[5,5],[5,5],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,2,5,5],2,[5,5],[5,5],[1,1],False,"NOTSET",[0,0,0,0])

            # Same but for a 2x2 kernel with stride of 1x1 (result is 3x3)
            if True:
                CreateConvolution([1,1,4,4],1,[2,2],[1,1],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,2,4,4],1,[2,2],[1,1],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,1,4,4],2,[2,2],[1,1],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,2,4,4],2,[2,2],[1,1],[1,1],False,"NOTSET",[0,0,0,0])

            # Different sized kernels
            if False:
                CreateConvolution([1,1,2,3],1,[2,3],[1,1],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,1,3,2],1,[3,2],[1,1],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,1,4,9],1,[2,3],[1,1],[1,1],False,"NOTSET",[0,0,0,0])
                CreateConvolution([1,1,9,4],1,[3,2],[1,1],[1,1],False,"NOTSET",[0,0,0,0])

            # Bigger more realistic examples
            if False:
                CreateConvolution([1,3,16,16],16,[2,2],[2,2],[1,1],False,"NOTSET",[0,0,0,0])

        # Left pad
        if True:
            CreateConvolution([1,1,2,2],1,[3,3],[1,1],[1,1],False,"NOTSET",[1,1,0,0])
            CreateConvolution([1,1,2,2],1,[3,3],[1,1],[1,1],False,"NOTSET",[0,0,1,1])

            #CreateConvolution([1,1,3,2],1,[3,3],[1,1],[1,1],False,"NOTSET",[1,0,0,0])

            #CreateConvolution([1,1,2,2],1,[3,3],[1,1],[1,1],False,"NOTSET",[0,0,1,1])


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
