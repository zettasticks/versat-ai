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

import sys
import numpy as np
import onnxruntime as ort
import os

@dataclass
class Test:
    leftShape: list[int] = None
    rightShape: list[int] = None
    leftTensor: any = None
    rightTensor: any = None
    outputTensor: any = None
    node: any = None
    leftRandomArray: any = None
    rightRandomArray: any = None

tests: list[Test] = []

def CreateTest(leftShape,rightShape):
    global tests

    testIndex = len(tests)

    test = Test()
    test.leftShape = leftShape
    test.rightShape = rightShape

    test.leftTensor = make_tensor_value_info(f"X{testIndex}", TensorProto.FLOAT, leftShape)
    test.rightTensor = make_tensor_value_info(f"Y{testIndex}", TensorProto.FLOAT, rightShape)

    OUT = make_tensor_value_info(f"OUT{testIndex}", TensorProto.FLOAT, [])  # shape_inference handles dims for out
    test.outputTensor = OUT

    test.node = make_node(f"Add", [f"X{testIndex}", f"Y{testIndex}"], [f"OUT{testIndex}"])    

    test.leftRandomArray = np.random.randn(*leftShape).astype(np.float32)
    test.rightRandomArray = np.random.randn(*rightShape).astype(np.float32)

    tests.append(test)

CreateTest([4,2],[4,2])
CreateTest([2,4,6],[2,4,6])

allInputNodesAndValuesInOrder = []
for x in tests:
    allInputNodesAndValuesInOrder.append([x.leftTensor,x.leftRandomArray])
    allInputNodesAndValuesInOrder.append([x.rightTensor,x.rightRandomArray])

allNodes = [x.node for x in tests]
allInputNodes = [x[0] for x in allInputNodesAndValuesInOrder]
allOutputNodes = [x.outputTensor for x in tests]

graph = make_graph(allNodes, "simpleTest",allInputNodes,allOutputNodes)

onnx_model = make_model(graph, opset_imports=[make_opsetid("", 7)])
check_model(onnx_model)
onnx_model = version_converter.convert_version(onnx_model, 7)
check_model(onnx_model)
shaped = onnx.shape_inference.infer_shapes(onnx_model)
check_model(shaped)

outputPath = sys.argv[1]
try:
    os.makedirs(os.path.join(outputPath,"test_data_set_0"))
except FileNotFoundError:
    print(f"Error creating path for output: {outputPath}")
    sys.exit(0)
except FileExistsError:
    pass # Not a problem if folder already exists.

for i,nodeAndValue in enumerate(allInputNodesAndValuesInOrder):
    value = nodeAndValue[1]
    with open(os.path.join(outputPath,f"test_data_set_0/input_{i}.pb"),"wb") as f:
        asTensor = numpy_helper.from_array(value)
        f.write(asTensor.SerializeToString())

sess = ort.InferenceSession(shaped.SerializeToString())

modelInputs = {}
for i,test in enumerate(tests):
    modelInputs[f"X{i}"] = test.leftRandomArray
    modelInputs[f"Y{i}"] = test.rightRandomArray

modelOutput = sess.run(None, modelInputs)

for i in range(len(tests)):
    with open(os.path.join(outputPath,f"test_data_set_0/output_{i}.pb"),"wb") as f:
        asTensor = numpy_helper.from_array(modelOutput[i])
        f.write(asTensor.SerializeToString())

save_onnx_model(shaped, os.path.join(outputPath,"model.onnx"))
