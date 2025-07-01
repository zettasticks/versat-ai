# Converts a given model into an equivalent model that outputs all the intermediate results.

import sys
import os
import glob

# Missing split_complex_to_pairs

from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model

import numpy as np
import onnx

import onnxruntime as ort

from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from onnx import numpy_helper

import matplotlib.pyplot as plt


def OnnxRenameNode(model, originalName, newName):
    for i in range(len(model.graph.node)):
        for j in range(len(model.graph.node[i].input)):
            if model.graph.node[i].input[j] == originalName:
                model.graph.node[i].input[j] = newName

        for j in range(len(model.graph.node[i].output)):
            if model.graph.node[i].output[j] == originalName:
                model.graph.node[i].output[j] = newName

    for i in range(len(model.graph.input)):
        if model.graph.input[i].name == originalName:
            model.graph.input[i].name = newName

    for i in range(len(model.graph.output)):
        if model.graph.output[i].name == originalName:
            model.graph.output[i].name = newName

    return model


def AddOutputsToEachNode(model):
    onnx.checker.check_model(model)

    shape = onnx.shape_inference.infer_shapes(model)
    nodeNameToOutputTensor = {}
    for index, node in enumerate(shape.graph.value_info):
        nodeNameToOutputTensor[node.name] = node.type.tensor_type

    sess = ort.InferenceSession(shape.SerializeToString())
    modelProperOutputs = [x.name for x in sess.get_outputs()]

    for output in modelProperOutputs:
        if output in nodeNameToOutputTensor:
            del nodeNameToOutputTensor[output]

    nodesToAdd = []
    for index, node in enumerate(model.graph.node):
        if node.output[0] in nodeNameToOutputTensor:
            nodesToAdd.append(node.output[0])

    for name in modelProperOutputs:
        nodesToAdd.append(name)

    newModel = select_model_inputs_outputs(model, outputs=nodesToAdd)

    for index, name in enumerate(nodesToAdd):
        # We do not want to rename the actual outputs, we still want to be able to test this model exactly as it started
        if name in modelProperOutputs:
            continue
        newModel = OnnxRenameNode(newModel, name, f"INTERMEDIATE_{index}")

    shaped = onnx.shape_inference.infer_shapes(newModel)
    return shaped


if __name__ == "__main__":
    model = onnx.load(sys.argv[1])
    result = AddOutputsToEachNode(model)
    save_onnx_model(result, sys.argv[2])
