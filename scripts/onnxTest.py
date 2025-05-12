import sys
import os
import glob

import numpy as np
import onnx

import onnxruntime as ort

from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from onnx import numpy_helper

import matplotlib.pyplot as plt
from onnx.tools.net_drawer import GetOpNodeProducer, GetPydotGraph

# TODO: Try 'from onnx.reference import ReferenceEvaluator'

print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")

#testDir = "../tests/caffenet-12-int8"
#modelName = "caffenet-12-int8.onnx"

#testDir = "../tests/tiny-yolov3_simple"
#modelName = "yolov3-tiny_simple.onnx"

testDir = "../tests/mnist_v7"
modelName = "model.onnx"

testModel = os.path.join(testDir,modelName)
testDataDir = os.path.join(testDir,"test_data_set_0")

model = onnx.load(testModel)
onnx.checker.check_model(model)
#print(model)

# Draw graph
pydot_graph = GetPydotGraph(
    model.graph, name=model.graph.name, rankdir="LR", node_producer=GetOpNodeProducer("docstring")
)
pydot_graph.write_dot("graph.dot")

# Perform inference
sess = ort.InferenceSession(testModel)
for x in sess.get_inputs():
   print(x.name)

inputs = []
inputs_num = len(glob.glob(os.path.join(testDataDir, 'input_*.pb')))
for i in range(inputs_num):
    input_file = os.path.join(testDataDir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(testDataDir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(testDataDir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

modelInputs = {x.name : y for x,y in zip(sess.get_inputs(),inputs)}

modelOutput = sess.run(None,modelInputs)

for ref_o, o in zip(ref_outputs, modelOutput):
    np.testing.assert_almost_equal(ref_o, o,decimal = 3)

#sys.exit(0)

# Input for the model (data to perform inference)
for name in sess.get_inputs():
   print("INPUT",name)
print()

# Outputs
for name in model.graph.output:
   print("OUTPUT",name)
print()

def PrettyDim(tensor):
   values = [str(x.dim_value) for x in tensor.shape.dim]
   return "x".join(values)

# All the data that is included in the model (weights,scales, etc)
for init in model.graph.initializer:
   print("PARAM",init.name,end=' ')
   print(init.dims)

shape = onnx.shape_inference.infer_shapes(model)

#for init in shape.graph.initializer:
#print("PARAM",init.name)
#print(type(init))
#sys.exit(0)

# Only contains the names for the outputs, I think
nodeNameToOutputTensor = {}
nodeIndexToOutputTensor = [None] * len(shape.graph.value_info)

for index,node in enumerate(shape.graph.value_info):
   nodeNameToOutputTensor[node.name] = node.type.tensor_type
   nodeIndexToOutputTensor[index] = node.type.tensor_type

# Also add the tensor type of the graph output
for node in sess.get_inputs():
   print("Input",node.shape)

for node in model.graph.output:
   nodeNameToOutputTensor[node.name] = node.type.tensor_type

for index,node in enumerate(model.graph.node):
   if(node.output[0] in nodeNameToOutputTensor):
      print(node.op_type,end=' ')
      print(node.output[0],end=' ')
      print(PrettyDim(nodeNameToOutputTensor[node.output[0]]))
   else:
      print(node.output[0]," not found")
