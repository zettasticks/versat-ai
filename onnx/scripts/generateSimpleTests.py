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

import numpy as np
import onnxruntime as ort

# [4,2] is the shape of the operations
X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 2])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [4, 2])

OUT = make_tensor_value_info(
    "OUT", TensorProto.FLOAT, None
)  # shape_inference handles dims for out
node = make_node("Add", ["X", "Y"], ["OUT"])
graph = make_graph(
    [node], "simpleTest", [X, Y], [OUT]  # nodes  # a name  # inputs
)  # outputs

onnx_model = make_model(graph, opset_imports=[make_opsetid("", 7)])
onnx_model = version_converter.convert_version(onnx_model, 7)
shaped = onnx.shape_inference.infer_shapes(onnx_model)
check_model(shaped)

x = np.random.randn(4, 2).astype(np.float32)
y = np.random.randn(4, 2).astype(np.float32)

with open("input_0.pb", "wb") as f:
    asTensor = numpy_helper.from_array(x)
    f.write(asTensor.SerializeToString())

with open("input_1.pb", "wb") as f:
    asTensor = numpy_helper.from_array(y)
    f.write(asTensor.SerializeToString())

sess = ort.InferenceSession(shaped.SerializeToString())
modelInputs = {"X": x, "Y": y}
modelOutput = sess.run(None, modelInputs)

with open("output_0.pb", "wb") as f:
    asTensor = numpy_helper.from_array(modelOutput[0])
    f.write(asTensor.SerializeToString())

save_onnx_model(shaped, "simpleAdd.onnx")
