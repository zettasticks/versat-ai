import sys
import os
import glob

# Missing split_complex_to_pairs

from onnx import shape_inference
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from dataclasses import dataclass

import pulp # TODO: Not sure how stable this is. Helpful for now, later we will see

import struct
import numpy as np
import onnx

import onnxruntime as ort

from enum import Enum,auto
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from onnx import numpy_helper

import itertools
import matplotlib.pyplot as plt

# TODO: Try 'from onnx.reference import ReferenceEvaluator'

print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")

#testDir = "../tests/caffenet-12-int8"
#modelName = "caffenet-12-int8.onnx"

#testDir = "../tests/tiny-yolov3_simple"
#modelName = "yolov3-tiny_simple.onnx"

testDir = "../tests/mnist_v7"
modelName = "model.onnx"

#testModel = os.path.join(testDir,modelName)
testModel = "test.onnx" # Testing the changed model
testDataDir = os.path.join(testDir,"test_data_set_0")

model = onnx.load(testModel)
onnx.checker.check_model(model)

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

isIntermediate = [False] * len(sess.get_outputs())

for index,output in enumerate(sess.get_outputs()):
   if("INTERMEDIATE" in output.name):
      isIntermediate[index] = True

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

print(len(modelOutput))

properOutputs = []
for index,output in enumerate(modelOutput):
   if(not isIntermediate[index]):
      properOutputs.append(output)

print(len(modelOutput))
print(len(properOutputs))
print(isIntermediate)
print([x.name for x in sess.get_outputs()])
outputNameToExpectedOutputNPArray = {x.name:y for x,y in zip(sess.get_outputs(),modelOutput)}

for ref_o, o in zip(ref_outputs, properOutputs):
   print("Gonna check if the test is good")
   np.testing.assert_almost_equal(ref_o, o,decimal = 9)

# Nodes have either inputs from other nodes or tensors, which are the constant values embedded in the model.
# If a given node input is a tensor then it is not an input and vice versa.
def GetTensor(model,tensorName):
   for tensor in model.graph.initializer:
      if(tensor.name == tensorName):
         return tensor

def GetShape(model,name):
   assert(name) # Make sure that we got a name, onnx models contain a lot of members that contain optional names, which might work for some models and not others. Care

   for value in model.graph.output:
      if(value.name == name):
         return [x.dim_value for x in value.type.tensor_type.shape.dim]
   for value in model.graph.input:
      if(value.name == name):
         return [x.dim_value for x in value.type.tensor_type.shape.dim]
   for value in model.graph.value_info:
      if(value.name == name):
         return [x.dim_value for x in value.type.tensor_type.shape.dim]

   # NOTE: We want this function to be able to obtain all the shapes from a given name.
   #       Need to care with the fact that some onnx names are optional. All the names that this function check are mandatory
   #       so this function works fine, just need to make sure that the name that we receive as input actually exists,
   #       and that we did not get it from a optional location, since different graphs might not implement them and this will fail.
   print(f"Could not find shape for {name}")
   assert(false)

shaped = shape_inference.infer_shapes(model)

if False:
   print(len(model.graph.value_info))
   for val in model.graph.value_info:
      print(val.name)
   for val in model.graph.input:
      print(val.name)
   for val in model.graph.output:
      print(val.name)
   sys.exit()

class Endianess(Enum):
   NATIVE = auto()
   LITTLE_ENDIAN = auto()
   BIG_ENDIAN = auto()

def PackNPArrayForCComsuption(array,endianess : Endianess = Endianess.NATIVE):
   endianArg = ""
   if(endianess == Endianess.NATIVE):
      endianArg = "@"
   elif(endianess == Endianess.LITTLE_ENDIAN):
      endianArg = "<"
   elif(endianess == Endianess.BIG_ENDIAN):
      endianArg = ">"
   else:
      assert(False)

   dims = len(array.shape)

   header = struct.Struct(f"{endianArg}i")

   # TODO: Faster implementation using buffers. We are processing a lot of data here, slow impl here is not advised.
   headerContent = header.pack(dims)
   for dim in array.shape:
      headerContent += header.pack(dim)

   data = struct.Struct(f"{endianArg}f")
   dataContent = bytes()
   for x in np.nditer(array):
      dataContent += data.pack(x)

   return headerContent + dataContent

@dataclass
class PackedArray:
   name: str
   data: bytes 

   def __repr__(self):
      return f"[PackedArray] {self.name} ({len(self.data)} bytes)"

class DataSourceType(Enum):
   INPUT = auto()
   INITIALIZER = auto()

@dataclass
class DataSource:
   sourceType : DataSourceType
   sourceName : str

@dataclass
class Operation:
   # Data extracted from the model
   nodeName: str
   opName: str
   inputs: list[DataSource]
   output: str # For now we are assuming that nodes only contain one output. Most graphs appear to follow this principle, even if the output is used by multiple nodes, the node itself only appaears to contain one. Maybe more exotic operations shatter this notion but will deal with them when they appear.
#   isOutputFinal: bool
   outputDimensions: list[int]
   testExpectedOutput: PackedArray = None
   #TODO: Eventually implement attributes properly. 

   # Data computed afterwards. Easier to store here
   outputMemoryAddress: int = 0 # Address at runtime. We precalculate it here to hopefully save memory, since embedded systems do not contain much

@dataclass
class CModelRepr:
   initializers: list[PackedArray]
   operations: list[Operation]

@dataclass
class MemoryAllocation:
   firstCycle: int
   lastCycle: int
   amount: int

cModel = CModelRepr([],[])

# Extract all the data that we care about from the graph into a simpler struct for further processing.
for node in model.graph.node:
   print("Node",node.name,":")
   print("  Attributes:")
   for attribute in node.attribute:
      print("    ",attribute,end='')

   print("Output:",node.output)

   print("  Graph inputs:")
   for name in node.input:
      tensor = GetTensor(model,name)
      if(not tensor):
         print("    ",name,GetShape(model,name))

   print("  Constant inputs:")
   for name in node.input:
      tensor = GetTensor(model,name)
      if(tensor):
         print("    ",name,tensor.dims)

         asNpArray = onnx.numpy_helper.to_array(tensor)
         packedBytes = PackNPArrayForCComsuption(asNpArray)
         cModel.initializers.append(PackedArray(name,packedBytes))

   outputDimensions = None
   print("  Output shapes")
   for output in node.output:
      shape = GetShape(shaped,output)
      outputDimensions = shape
      print("    ",shape,end=' ')
   print()


   dataSources = []
   for name in node.input:
      tensor = GetTensor(model,name)
      source = None
      if(tensor):
         source = DataSource(DataSourceType.INITIALIZER,name) 
      else:         
         source = DataSource(DataSourceType.INPUT,name) 
      dataSources.append(source)

   outputName = node.output[0] # Can a node have more than one output? 
   expectedOutputAsNPArray = outputNameToExpectedOutputNPArray[outputName]
   packedExpectedOutput = PackNPArrayForCComsuption(expectedOutputAsNPArray)

   outputTest = PackedArray(outputName,packedExpectedOutput)

   op = Operation(node.name,node.op_type,dataSources,outputName,outputDimensions,outputTest)
   cModel.operations.append(op)

   #print(cModel)

for c in cModel.operations:
   print(c)
   print()

# Calculate memory required and memory allocation model.
def IndexOfNodesThatUseOutput(cModel,outputName):
   indexes = []
   for index,op in enumerate(cModel.operations):
      for inp in op.inputs:
         if(inp.sourceName == outputName):
            indexes.append(index)
   return indexes

# Heavy weight algorithm to calculate the best way of performing memory allocations.
# We reduce the problem to rectangle fitting. The memory allocation is represented as rectangles where
# the width is the amount of memory used and the height is the "time" where the allocation occurs
# Each onnx operation advances time by 1 unit.
# We encode the problem as a Integer Linear program and use a solver to find the optimal solution
def CalculateOptimalMemoryAllocationOffset(memoryAllocations: list[MemoryAllocation]):
   M = 1000000 # Not a big fan of this trick. What if the memory size becomes bigger than this number?
   # TODO: Need to find an alternative

   # Create all the variables that we are gonna use
   z = {}
   r_x = [None] * len(memoryAllocations)

   for index,mem in enumerate(memoryAllocations):
      r_x[index]  = pulp.LpVariable(f"r{index}_x",lowBound=0,cat='Integer')

   for memAndIndex in itertools.combinations(enumerate(memoryAllocations),2):
      i,leftMem = memAndIndex[0]
      k,rightMem = memAndIndex[1]

      z[f"{i}_{k}_0"] = pulp.LpVariable(f"{i}_{k}_0",0,1,cat='Integer')
      z[f"{i}_{k}_1"] = pulp.LpVariable(f"{i}_{k}_1",0,1,cat='Integer')
      z[f"{i}_{k}_2"] = pulp.LpVariable(f"{i}_{k}_2",0,1,cat='Integer')
      z[f"{i}_{k}_3"] = pulp.LpVariable(f"{i}_{k}_3",0,1,cat='Integer')

   H = pulp.LpVariable("H",lowBound=0)

   prob = pulp.LpProblem("myProblem", pulp.LpMinimize)

   for memAndIndex in itertools.combinations(enumerate(memoryAllocations),2):
      i,leftMem = memAndIndex[0]
      k,rightMem = memAndIndex[1]

      r0_x = r_x[i]
      r1_x = r_x[k]

      r0_w = leftMem.amount
      r1_w = rightMem.amount

      r0_y = leftMem.firstCycle
      r1_y = rightMem.firstCycle

      r0_l = leftMem.lastCycle
      r1_l = rightMem.lastCycle

      z_0 = z[f"{i}_{k}_0"]
      z_1 = z[f"{i}_{k}_1"]
      z_2 = z[f"{i}_{k}_2"]
      z_3 = z[f"{i}_{k}_3"]

      prob += r1_x + r1_w <= r0_x + M*z_0
      prob += r0_x + r0_w <= r1_x + M*z_1
      prob += r1_l        <= r0_y + M*z_2
      prob += r0_l        <= r1_y + M*z_3
      prob += z_0 + z_1 + z_2 + z_3 <= 3

   # Objective to minimize as a constraint in a dummy variable H
   for index,mem in enumerate(memoryAllocations):
      x = r_x[index]
      w = mem.amount
      prob += x + w <= H

   # True objective
   prob += H

   status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

   # TODO: Can this occur?
   if(not status):
      print("Problem calculating the optimal memory allocations")
      return None

   offsets = []
   for x in r_x:
      offsets.append(int(pulp.value(x)))

   totalMemoryNeeded = int(pulp.value(H))

   return totalMemoryNeeded,offsets

memoryAllocations = []
for index,c in enumerate(cModel.operations):
   indexes = IndexOfNodesThatUseOutput(cModel,c.output)

   if(not indexes):
      lastCycle = len(cModel.operations)
   else:
      lastCycle = max(indexes)

   # In order to prevent operations that write on top of their input
   lastCycle += 1

   # Very simple memory calculation, might be wrong, especially if we decide to add padding and stuff like that. Need to make this more customizable.
   memoryRequired = 4  # Size of a float
   for dim in c.outputDimensions:
      memoryRequired *= dim

   memoryAllocations.append(MemoryAllocation(index,lastCycle,memoryRequired))

totalMemoryNeeded,offsets = CalculateOptimalMemoryAllocationOffset(memoryAllocations)
print("total memory needed:",totalMemoryNeeded)
print(offsets)
