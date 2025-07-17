import sys

sys.path.append("./scripts")
from onnxMain import GenerateDebug, GenerateModelFromOnnxModel

GenerateDebug(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
