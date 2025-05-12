# Converts a given model into an equivalent model that outputs all the intermediate results.

import sys
import onnx
from onnx import helper

model = onnx.load(sys.argv[0])

if not model:
   print("Error loading module")
   sys.exit(0)

for node in model.graph.node:
   print(node.output)
   #helper.make_tensor_value_info(name=node.name,elem_type=onnx.TensorProto.FLOAT,shape=) 