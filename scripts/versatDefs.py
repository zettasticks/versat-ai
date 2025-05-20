from dataclasses import dataclass,field
from enum import Enum,auto

import numpy as np

@dataclass
class PackedArrays:
   data: bytes
   offsets: list[int]

class Endianess(Enum):
   NATIVE = auto()
   LITTLE_ENDIAN = auto()
   BIG_ENDIAN = auto()

class DataSourceType(Enum):
   MODEL_INPUT = auto()
   NODE_INPUT = auto()
   INITIALIZER = auto()

@dataclass
class DataSource:
   sourceType : DataSourceType
   name : str

   # Computed afterwards. Easier to store directly.
   index : int = -1
   correctInputIndex: int = -1

class MemoryType(Enum):
   TEMP = auto()
   OUTPUT = auto()

@dataclass
class MemoryLocation:
   offset : int
   memType : MemoryType

@dataclass
class Operation:
   # Data extracted from the model
   nodeName: str
   opName: str
   inputs: list[DataSource]
   output: str # For now we are assuming that nodes only contain one output. Most graphs appear to follow this principle, even if the output is used by multiple nodes, the node itself only appaears to contain one. Maybe more exotic operations shatter this notion but will deal with them when they appear.
   inputDimensions: list[list[int]]
   outputDimensions: list[int]

   #TODO: Eventually implement attributes properly. 

   # Data computed from extracted model. 
   outputMemoryAddress: MemoryLocation = None # Address at runtime. We precalculate it, we do not allocate memory at runtime.

# A more useful representation for our use cases than having to interact with the onnx model directly
@dataclass
class Model:
   onnxModel: any # Already transformed to ouput intermediate results
   sess: any = None
   modelOutputs: list[str] = None
   isModelOutputIntermediate: list[bool] = None
   modelInputs : list[str] = field(default_factory=list)
   initializers: list[np.array] = field(default_factory=list)
   operations: list[Operation] = field(default_factory=list)
   tempMemoryNeeded: int = -1
   outputMemoryNeeded: int = -1

@dataclass
class ModelRunResult:
   outputs: list[np.array] # Maps operation to its output. 

@dataclass
class MemoryAllocation:
   firstCycle: int
   lastCycle: int
   amount: int