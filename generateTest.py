import sys

sys.path.append("./onnx/scripts")
from onnxTest import GenerateDebug, GenerateModelFromOnnxModel

GenerateDebug(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
