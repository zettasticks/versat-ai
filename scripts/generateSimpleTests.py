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


# CreateBinaryOpTest("Add", [1], [1])
# CreateBinaryOpTest("Add", [4], [4])

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

    CreateUnaryOpTest("Relu", [1])
    CreateUnaryOpTest("Relu", [4])
    CreateUnaryOpTest("Relu", [2, 4])
    CreateUnaryOpTest("Relu", [2, 4, 6])
    CreateUnaryOpTest("Relu", [2, 4, 6, 8])

    CreateReshapeTest([4, 2], [8])

if True:
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
