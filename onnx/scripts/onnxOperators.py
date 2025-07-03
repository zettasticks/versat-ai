from typing import Callable
from dataclasses import dataclass, field
from copy import copy

from versatDefs import (
    Operation,
    InstantiatedAttribute,
    OnnxAttribute,
    OnnxAttributeType,
    OnnxOperatorSpec,
)
from enum import Enum, auto

# TODO: I eventually want to start generating the C structs from the emitters defined here. Really clubersome to have to match
#      the emitter code with the C code, any change requires to carefully interact with

# TODO: Because Onnx supports variable sized tensors, we might want to start moving the tensors shape calculations to runtime.
#      For now this is fine because most models do not have dynamic tensor shapes, but eventually need to do this.


def ExtendShape(shapeList, dimensions):
    res = copy(shapeList)
    if len(shapeList) < dimensions:
        for i in range(dimensions - len(shapeList)):
            res = [1] + res
    return res


def BroadCastShape(op0, op1):
    if len(op0) != len(op1):
        length = max(len(op0), len(op1))
        op0 = ExtendShape(op0)
        op1 = ExtendShape(op1)

    res = []
    for a, b in zip(op0, op1):
        res.append(max(a, b))
    return res


def MakeAttrBoundedString(allowedStringValues: list[str], default: str = None):
    return OnnxAttribute(OnnxAttributeType.BOUNDED_STRING, allowedStringValues, default)


def MakeAttrBoundedInteger(allowedIntegerValues: list[int], default: int = None):
    return OnnxAttribute(
        OnnxAttributeType.BOUNDED_INTEGER, allowedIntegerValues, default
    )


def MakeAttrIntegerList(defaultValue):
    return OnnxAttribute(OnnxAttributeType.INTEGER_LIST, [], defaultValue)


def MakeAttrIngeger(defaultValue):
    return OnnxAttribute(OnnxAttributeType.INTEGER, [], defaultValue)


# Some attributes have defaults that depend on the operator (like the size of the spatial axis and such)
# This function essentially instantiates default values such that outer code does not have to check if an attributes exists or not.
def GetAttributesForOperator(op: Operation) -> dict[str, InstantiatedAttribute]:
    opName = op.opName

    if not opName in operatorNameToSpec:
        return None

    opSpec = operatorNameToSpec[opName]

    # TODO: We can actually remove this and the function. All we need is to use the data that we have to figure out whether we need to use a default attribute value or a value that is contained inside the operator.

    if opSpec.attributesForOperatorFunction:
        return opSpec.attributesForOperatorFunction(op)
    else:
        # TODO: Need to instantiate the attributes with their default values.
        pass
        # return opSpec.attributesDict


def EmitAdd(emitter, op: Operation):
    maxDims = max(len(op.inputDimensions[0]), len(op.inputDimensions[1]))

    op0 = ExtendShape(op.inputDimensions[0], maxDims)
    op1 = ExtendShape(op.inputDimensions[1], maxDims)

    broadCastedShape = BroadCastShape(op0, op1)

    aux_0 = emitter.EmitArray("int64_t", op0)
    aux_1 = emitter.EmitArray("int64_t", op1)
    aux_2 = emitter.EmitArray("int64_t", broadCastedShape)

    return [maxDims, aux_0, aux_1, aux_2]


def EmitRelu(emitter, op: Operation):
    aux = emitter.EmitArray("int64_t", op.inputDimensions[0])
    dims = len(op.inputDimensions[0])
    return [dims, aux]


def EmitMaxPool(emitter, op: Operation):
    dims = len(op.inputDimensions[0])
    inputShape = emitter.EmitArray("int64_t", op.inputDimensions[0])
    outputShape = emitter.EmitArray("int64_t", op.outputDimensions)

    kernel_shape = op.parsedAttributes["kernel_shape"].value

    print(kernel_shape)

    return [dims, inputShape, outputShape, kernel_shape[0], kernel_shape[1]]


def EmitConv(emitter, op: Operation):
    inputDim = len(op.inputDimensions[0])
    inputShape = emitter.EmitArray("int64_t", op.inputDimensions[0])

    kernelDim = len(op.inputDimensions[1])
    kernelShape = emitter.EmitArray("int64_t", op.inputDimensions[1])

    outDim = len(op.outputDimensions)
    outShape = emitter.EmitArray("int64_t", op.outputDimensions)

    return [inputDim, inputShape, kernelDim, kernelShape, outDim, outShape]


def EmitReshape(emitter, op: Operation):
    op0 = op.inputDimensions[0]
    dimIn = len(op.inputDimensions[0])
    dimOut = op.inputDimensions[1][0]

    aux_0 = emitter.EmitArray("int64_t", op0)

    return [aux_0, dimIn, dimOut]


def EmitMatMul(emitter, op: Operation):
    op0 = op.inputDimensions[0]
    op1 = op.inputDimensions[1]
    res = [op0[0], op1[1]]

    aux_0 = emitter.EmitArray("int64_t", op0)
    aux_1 = emitter.EmitArray("int64_t", op1)
    aux_2 = emitter.EmitArray("int64_t", res)

    return [aux_0, len(op0), aux_1, len(op1), aux_2, len(res)]


def IsOperatorRegistered(opName: str):
    return opName in operatorNameToSpec


def EmitParameterList(emitter, op: Operation):
    global operatorNameToSpec
    spec = operatorNameToSpec.get(op.opName, None)

    if not spec:
        print(
            f"Operator {op.opName} is not registered and no implementation exists for it"
        )
        print(f"Know operators: {operatorNameToSpec.keys()}")
    elif spec.emitFunction:
        return spec.emitFunction(emitter, op)
    else:
        return "{}"


convAttributes = {
    "auto_pad": MakeAttrBoundedString(
        ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"], "NOTSET"
    ),
    "dilations": MakeAttrIntegerList(1),
    "group": MakeAttrIngeger(1),
    "kernel_shape": MakeAttrIntegerList(None),
    "pads": MakeAttrIntegerList(0),
    "strides": MakeAttrIntegerList(1),
}


# TODO: This functions only exist because we are not properly parsing the attributes when parsing the model.
#       Otherwise we could easily make this function in a generic way.
def ConvAttributesForOperation(op: Operation) -> dict[str, InstantiatedAttribute]:
    global convAttributes

    res = {}
    for name, attrType in convAttributes.items():
        if name in op.parsedAttributes:
            res[name] = op.parsedAttributes[name]
        else:
            res[name] = InstantiatedAttribute(attrType, attrType.defaultValue)

    return res


maxPoolAttributes = {
    "auto_pad": MakeAttrBoundedString(
        ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"], "NOTSET"
    ),
    "ceil_mode": MakeAttrIngeger(0),
    "dilations": MakeAttrIntegerList(1),
    "kernel_shape": MakeAttrIntegerList(None),
    "pads": MakeAttrIntegerList(0),
    "storage_order": MakeAttrBoundedInteger([0, 1], 0),
    "strides": MakeAttrIntegerList(1),
}


def MaxPoolAttributesForOperation(op: Operation) -> dict[str, InstantiatedAttribute]:
    global maxPoolAttributes

    res = {}
    for name, attrType in maxPoolAttributes.items():
        if name in op.parsedAttributes:
            res[name] = op.parsedAttributes[name]
        else:
            res[name] = InstantiatedAttribute(attrType, attrType.defaultValue)

    return res


def GetOperatorSpec(opName):
    global operatorNameToSpec
    return operatorNameToSpec[opName]


# Register new operators here
operatorNameToSpec = {}
operatorNameToSpec["Add"] = OnnxOperatorSpec("Add", EmitAdd, True, False, [], [], True)
operatorNameToSpec["Conv"] = OnnxOperatorSpec(
    "Conv", EmitConv, False, False, convAttributes, ConvAttributesForOperation
)
operatorNameToSpec["Relu"] = OnnxOperatorSpec("Relu", EmitRelu, False, False)
operatorNameToSpec["MaxPool"] = OnnxOperatorSpec(
    "MaxPool",
    EmitMaxPool,
    False,
    False,
    maxPoolAttributes,
    MaxPoolAttributesForOperation,
)

# Care, not fully defined, mostly to stop key errors from appearing
operatorNameToSpec["Reshape"] = OnnxOperatorSpec("Reshape", EmitReshape, False, False)
operatorNameToSpec["MatMul"] = OnnxOperatorSpec("MatMul ", EmitMatMul, False, False)
