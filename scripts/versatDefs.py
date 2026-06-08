from typing import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pprint import pformat

import numpy as np

@dataclass
class PackedArrays:
    data: bytes
    offsets: list[int]


class Endianess(Enum):
    NATIVE = auto()  # NATIVE
    LITTLE_ENDIAN = auto()
    BIG_ENDIAN = auto()


class DataSourceType(Enum):
    MODEL_INPUT = auto()
    NODE_INPUT = auto()
    INITIALIZER = auto()


@dataclass
class CDataHandle:
    index: int


@dataclass
class TypedArray:
    dtype: str
    data: list[any]
    name: str = None

# A source of data. Initializer data is stored in here. ModelInputs data for validation are also stored.
# Node data only contains the index of the outputting node. Valid data can be obtained by getting node
# and getting the correctOutputData
@dataclass
class DataSource:
    sourceType: DataSourceType
    name: str # Only used during graph construction to bridge between onnx data and ours. Could probably remove this.

    index: int = -1
    data: np.array = None
    tensorDims: list[int | str] | None = None  # Not always guaranteed to exist.

    def __repr__(self):
        if (self.sourceType == DataSourceType.MODEL_INPUT and self.data is not None):
            return self.sourceType.name + "_" + str(self.index) + "_" + str(self.data.shape)

        return self.sourceType.name + " " + str(self.index)

def MakeNodeInput(name,nodeIndex: int):
    out = DataSource(DataSourceType.NODE_INPUT,name,nodeIndex)
    return out
    
def CopyDataSource(inp: DataSource):
    dataCopy = None
    if (inp.data is not None):
        dataCopy = np.copy(inp.data)
    tensorDims = None
    if (inp.tensorDims is not None):
        tensorDims = list(inp.tensorDims)

    out = DataSource(inp.sourceType,inp.name,inp.index,dataCopy,tensorDims)
    return out
    
class MemoryType(Enum):
    TEMP = auto()
    OUTPUT = auto()


@dataclass
class MemoryLocation:
    offset: int
    memType: MemoryType


class OnnxAttributeType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    BOUNDED_INTEGER = auto()
    INTEGER_LIST = auto()
    AXIS_LIST = auto()
    AXIS_PAIR_LIST = auto()
    BOUNDED_STRING = auto()
    ENUM = auto()


class PaddingType(Enum):
    NOTSET = 0
    SAME_UPPER = 1
    SAME_LOWER = 2
    VALID = 3


@dataclass
class OnnxAttribute:
    attrType: OnnxAttributeType
    allowedValues: list[any]
    defaultValue: any

# Graph stuff: Every operation (a node in the graph) contains an index
# that is not guaranteed to match with the position of the node in the
# graph array.  This index is the value that is stored inside the data
# sources. Because graph operations can add and remove nodes, this
# makes sure that we do not need to update every node everytime we
# make a change.
# For model inputs and initializers the index is the position on the array
# since these do not change or are removed.

@dataclass
class Operation:
    # Data extracted from the model
    nodeName: str
    opName: str
    nodeIndex: int
    inputs: list[DataSource]
    outputName: str  # For now we are assuming that nodes only contain one output. Most graphs appear to follow this principle, even if the output is used by multiple nodes, the node itself only appears to contain one. Maybe more exotic operations shatter this notion but will deal with them when they appear.
    inputDimensions: list[list[int | str]]
    outputDimensions: list[list[int | str]]
    parsedAttributes: dict[str, any] = field(default_factory=dict)

    # Data used to check operation correctness (TODO: Only one since we only support 1 output per node right now)
    correctOutputData: np.array = None

    # Data computed after graph operations
    outputMemoryAddress: MemoryLocation = (
        None  # Address at runtime. We precalculate it, we do not allocate memory at runtime.
    )

    def __repr__(self):
        ev = vars(self)
        if self.correctOutputData is not None:
            ev = dict(ev)
            #ev['correctOutputData'] = self.correctOutputData.shape
        val = pformat(ev)
        return val
    
class BroadcastType(Enum):
    NO_BROADCAST = auto()
    UNIDIRECTIONAL = auto()
    MULTIDIRECTIONAL = auto()


@dataclass
class OnnxOperatorSpec:
    name: str
    index: int  # Must be different for each operator.
    emitFunction: Callable
    emitStructure: list = field(default_factory=list)
    attributesDict: dict[str, OnnxAttribute] = field(default_factory=dict)
    supportedByVersat: bool = False
    broadcastType: BroadcastType = BroadcastType.NO_BROADCAST
    floatPrecision: float = 0.001


@dataclass
class InputData:
    name: str
    shape: list[int | str]
    isOriginal: bool = True  # In order to extract data from nodes, we add custom output ports. This variable is true only for the ports that are original, no modification made

@dataclass
class Port:
    index: int
    port: int

@dataclass
class GenericDataSource:
    dims: list[int]
    data: np.array
    
# A more useful representation for our use cases than having to interact with the onnx model directly
@dataclass
class Model:
    # TODO: This might need to go. During graph operations we might change stuff and eventually have to
    modelInputs: list[InputData] = field(default_factory=list)
    
    operations: list[Operation] = field(default_factory=list)

    nextInitializerIndex: int = 0
    nextOperationIndex: int = 0
    
    # Calculated after parameter instantiation and other model operations that change operations are performed
    tempMemoryNeeded: int = -1
    outputMemoryNeeded: int = -1

    def GetGenericDataSource(self,src: DataSource):
        dims = None
        data = None
        
        if(src.sourceType == DataSourceType.NODE_INPUT):
            outputNode = self.GetOperationByIndexOrFail(src.index)
            data = outputNode.correctOutputData
            dims = list(outputNode.outputDimensions[0])
        else:
            data = src.data
            dims = src.tensorDims

        return GenericDataSource(dims,data)
    
    
    # If we already have A -> B:x and want to insert in the middle C (A -> C:y and C -> B:x)
    def InsertInTheMiddle(self,B: Operation,x:int, C: Operation,y: int):
        C.inputs[y] = CopyDataSource(B.inputs[x])

        if(C.opName == "Transpose"):
            # Stupid way of trying to preserve as much info as possible
            src = self.GetGenericDataSource(C.inputs[y])

            C.inputDimensions[0] = src.dims
            C.outputDimensions[0] = src.dims
            C.correctOutputData = np.copy(np.transpose(src.data))

        B.inputs[x] = MakeNodeInput(B.inputs[x].name,C.nodeIndex)

    # TODO: Input size is stupid
    def AddOperation(self,opType,inputSize):
        opIndex = self.NextOperationIndex()
        name = f"ADDED_{opIndex}"
        outputName = f"GENERATED_OUTPUT_{opIndex}"
        inputs = [None] * inputSize
        inputDims = [None] * inputSize
        outputDims = [None] * inputSize
        newOp = Operation(name,opType,opIndex,inputs,outputName,inputDims,outputDims)
        self.operations.append(newOp)

        return newOp
        
    def RemoveOperation(self,toRemove: Operation):
        for op in self.operations:
            for inp in op.inputs:
                if (inp.sourceType == DataSourceType.NODE_INPUT and inp.index == toRemove.nodeIndex):
                    inp.sourceType = DataSourceType.INITIALIZER
                    inp.index = self.NextInitializerIndex()
                    inp.data = toRemove.correctOutputData

        self.operations.remove(toRemove)
    
    def NextInitializerIndex(self):
        res = self.nextInitializerIndex
        self.nextInitializerIndex += 1
        return res

    def NextOperationIndex(self):
        res = self.nextOperationIndex
        self.nextOperationIndex += 1
        return res
    
    def GetOperationByIndexOrFail(self,index):
        for op in self.operations:
            if (op.nodeIndex == index):
                return op
        assert False

@dataclass
class ModelRunResult:
    outputs: list[np.array]  # Maps operation to its output.


@dataclass
class MemoryAllocation:
    firstCycle: int
    lastCycle: int
    amount: int
