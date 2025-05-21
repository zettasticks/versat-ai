import sys
import os
import glob

# Missing split_complex_to_pairs

from scripts.versatDefs import *
from scripts.memoryAllocator import CalculateMemoryAllocations
from scripts.onnxAddOutputsToIntermediate import AddOutputsToEachNode
from scripts.onnxOperators import EmitParameterList,IsOperatorRegistered
from copy import copy

from onnx import shape_inference
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from dataclasses import dataclass,field
from functools import reduce

import struct
import numpy as np
import onnx

import onnxruntime as ort

from enum import Enum,auto
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
from onnx import numpy_helper

import matplotlib.pyplot as plt

# Nodes have either inputs from other nodes or initializers, which are the constant values embedded in the model.
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

def EndianessToStructArg(endianess: Endianess):
   endianArg = ""
   if(endianess == Endianess.NATIVE):
      endianArg = "@"
   elif(endianess == Endianess.LITTLE_ENDIAN):
      endianArg = "<"
   elif(endianess == Endianess.BIG_ENDIAN):
      endianArg = ">"
   else:
      assert(False)
   
   return endianArg

def PackArrayNoHeader(array,endianess : Endianess = Endianess.NATIVE):
   endianArg = EndianessToStructArg(endianess)

   dtype = array.dtype
   formatArg = "f"
   if(dtype == np.int64):
      formatArg = "q"
   
   data = struct.Struct(f"{endianArg}{formatArg}")
   dataContent = bytearray()
   for x in np.nditer(array):
      dataContent += data.pack(x)

   return dataContent

def PackMultipleArrays(arrayList,endianess : Endianess = Endianess.NATIVE):
   data = bytearray()
   offsets = []
   for array in arrayList:
      offsets.append(len(data))
      data += PackArrayNoHeader(array)

   return PackedArrays(data,offsets)

def IndexOfNodeThatProducesOutput(cModel,outputName):
   for index,op in enumerate(cModel.operations):
      if(outputName == op.output):
         return index
   return None

def GenerateModelFromOnnxModel(onnxModel):
   shaped = shape_inference.infer_shapes(onnxModel)
   cModel = Model(shaped)
   
   inputNames = []
   for value in onnxModel.graph.input:
      if(not GetTensor(shaped,value.name)):
         inputNames.append(value.name)
   cModel.modelInputs = inputNames

   # Extract all the data that we care about from the graph into a simpler struct for further processing.
   for node in onnxModel.graph.node:
      attributes = {}
      for attribute in node.attribute:
         # TODO: Proper parsing and unpacking of attributes
         attributes[attribute.name] = attribute.ints

      inputDimensions = []
      for name in node.input:
         shape = GetShape(shaped,name)
         inputDimensions.append(shape)
         tensor = GetTensor(onnxModel,name)
         if(tensor):
            asNpArray = onnx.numpy_helper.to_array(tensor)
            cModel.initializers.append(asNpArray)

      outputDimensions = None
      for output in node.output:
         shape = GetShape(shaped,output)
         outputDimensions = shape

      dataSources = []
      for name in node.input:
         tensor = GetTensor(onnxModel,name)
         source = None
         if(tensor):
            source = DataSource(DataSourceType.INITIALIZER,name) 
         elif(name in cModel.modelInputs):
            source = DataSource(DataSourceType.MODEL_INPUT,name) 
         else:
            source = DataSource(DataSourceType.NODE_INPUT,name) 
         dataSources.append(source)

      outputName = node.output[0] # Can a node have more than one output? 

      op = Operation(node.name,node.op_type,dataSources,outputName,inputDimensions,outputDimensions,attributes)
      cModel.operations.append(op)

   CalculateMemoryAllocations(cModel)

   return cModel

def RunModel(model : Model,inputs):
   if(not model.sess):
      model.sess = ort.InferenceSession(model.onnxModel.SerializeToString())

   modelInputs = {x.name : y for x,y in zip(model.sess.get_inputs(),inputs)}
   modelOutput = model.sess.run(None,modelInputs)
   
   mappedOutputs = {x.name:y for x,y in zip(model.sess.get_outputs(),modelOutput)}
   outputs = [None] * len(model.operations)
   for index,op in enumerate(model.operations):
      outputs[index] = mappedOutputs[op.output]

   return ModelRunResult(outputs)

@dataclass
class CDataHandle:
   index: int

@dataclass
class TypedArray:
   dtype: str
   data: list[any]
   name: str = None

class CDataEmitter:
   def __init__(self):
      self.arrays = []
      self.namedArrays = []

   def EmitNamedArray(self,name,dtype,data):
      self.namedArrays.append(TypedArray(dtype,copy(data),name))

   def EmitArray(self,dtype,data):
      assert(isinstance(data,list))
      index = len(self.arrays)

      self.arrays.append(TypedArray(dtype,copy(data)))

      return CDataHandle(index)

   def Representation(self):
      content = ""

      def ItemRepr(item):
         if isinstance(item,CDataHandle):
            return f"temp_{item.index}"
         elif isinstance(item,list):
            return "{" + ",".join([ItemRepr(x) for x in item]) + "}"
         else:
            return str(item)

      for index,tarray in enumerate(self.arrays):
         dtype = tarray.dtype
         data = tarray.data

         content += f"{dtype} temp_{index}[] = " + "{" + ",".join(ItemRepr(x) for x in data) + "};\n"

      for index,tarray in enumerate(self.namedArrays):
         dtype = tarray.dtype
         data = tarray.data
         name = tarray.name
         amount = len(tarray.data)

         content += f"{dtype} {name}[{amount}] = " + "{" + ",".join(ItemRepr(x) for x in data) + "};\n"

      return content

def GenerateDebug(testLocation: str,outputLocation: str):
   #print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")

   testModelLocation = os.path.join(testLocation,"model.onnx")

   # TODO: Only fetching one test, for now
   testDataDir = os.path.join(testLocation,"test_data_set_0")

   model = onnx.load(testModelLocation)
   model = AddOutputsToEachNode(model)
   onnx.checker.check_model(model)

   # Perform inference
   sess = ort.InferenceSession(model.SerializeToString())

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

   properOutputs = []
   for index,output in enumerate(modelOutput):
      if(not isIntermediate[index]):
         properOutputs.append(output)

   for ref_o, o in zip(ref_outputs, properOutputs):
      np.testing.assert_almost_equal(ref_o, o,decimal = 9)

   cModel = GenerateModelFromOnnxModel(model)
   cModel.isModelOutputIntermediate = isIntermediate

   # TODO: Implement multiple testcases by running the model multiple times and outputting multiple correct data bins.
   # NOTE: Is it possible for different testcases to generate different amounts of correctData? It shouldn't be possible.
   result = RunModel(cModel,inputs)
   correctData = result.outputs

   outputNameToNodeIndex = {}
   ind = 0
   for index,c in enumerate(cModel.operations):
      outputNameToNodeIndex[c.output] = ind
      ind += 1

   packedCorrectData = PackMultipleArrays(correctData)
   packedInitializers = PackMultipleArrays(cModel.initializers)

   # Calculate initializer position
   initializersSeen = 0
   for index,c in enumerate(cModel.operations):
      for source in c.inputs:
         if(source.sourceType == DataSourceType.INITIALIZER):
            source.index = initializersSeen
            initializersSeen += 1
         elif(source.sourceType == DataSourceType.MODEL_INPUT):
            for index,name in enumerate(cModel.modelInputs):
               if(name == source.name):
                  source.index = index
         else:
            source.index = IndexOfNodeThatProducesOutput(cModel,source.name)
            source.correctInputIndex = outputNameToNodeIndex[source.name]

   print("#include \"versat_ai.h\"")
   print()
   print("size_t GetOutputMemorySize(){return ",cModel.outputMemoryNeeded,";}",sep='')
   print("size_t GetTemporaryMemorySize(){return ",cModel.tempMemoryNeeded,";}",sep='')
   print("size_t GetModelMemorySize(){return ",len(packedInitializers.data),";}",sep='')
   print("size_t GetCorrectMemorySize(){return ",len(packedCorrectData.data),";}",sep='')
   print()

   print(f"int numberLayers = {len(cModel.operations)};")

   layerInfo = []
   for index,c in enumerate(cModel.operations):
      outputSize = reduce(lambda x,y : x * y,c.outputDimensions) * 4 # Sizeof(float)
      layerInfo.append("{" + f"\"{c.nodeName}\",\"{c.opName}\",{outputSize}" + "}")

   print("LayerInfo layers[] = {",",".join(layerInfo),"};\n")

   opcodeToOperationList = {}
   for index,c in enumerate(cModel.operations):
      opcodeToOperationList[c.opName] = opcodeToOperationList.get(c.opName,[]) + [c]

   emitter = CDataEmitter()

   for opcode in opcodeToOperationList:
      if(not IsOperatorRegistered(opcode)):
         continue

      operationList = opcodeToOperationList[opcode]

      structs = []
      for index,op in enumerate(operationList):
         structs.append(EmitParameterList(emitter,op))

      emitter.EmitNamedArray(f"{opcode}Infos",f"{opcode}Info",structs)

   content = emitter.Representation()
   print(content)

   # Placeholder for operators not yet registered
   for opcode,opList in opcodeToOperationList.items():
      amount = len(opList)

      if(IsOperatorRegistered(opcode)):
         continue

      content = []
      for op in opList:
         content.append("{}")

      print(f"{opcode}Info {opcode}Infos[{amount}] = ","{",",".join(content),"};",sep='')

   print()
   print("InferenceOutput DebugRunInference(void* outputMemory,void* temporaryMemory,void** inputs,void* modelMemory,void* correctInput){")

   debugging = True
   opSeen = {}
   for index,c in enumerate(cModel.operations):
      content = []
      for inp in c.inputs:
         if(inp.sourceType == DataSourceType.INITIALIZER):
            content.append(f"OFFSET_PTR(modelMemory,{packedInitializers.offsets[inp.index]})")
         elif(inp.sourceType == DataSourceType.MODEL_INPUT):
            content.append(f"inputs[{inp.index}]")
         else:
            if(debugging):
               content.append(f"OFFSET_PTR(correctInput,{packedCorrectData.offsets[inp.index]})")
            else:
               content.append(f"res_{inp.index}")

      outputStr = ""
      if(c.outputMemoryAddress.memType == MemoryType.TEMP):
         outputStr = f"OFFSET_PTR(temporaryMemory,{c.outputMemoryAddress.offset})"
      else:
         outputStr = f"OFFSET_PTR(outputMemory,{c.outputMemoryAddress.offset})"

      opIndex = opSeen.get(c.opName,-1) + 1
      opSeen[c.opName] = opIndex

      print(f"  void* res_{index} = ",c.opName,'(',",".join(content),f",{outputStr},{index},&{c.opName}Infos[{opIndex}]);",sep='')
      if(debugging and (IsOperatorRegistered(c.opName))):
         print(f"  AssertAlmostEqual(res_{index},OFFSET_PTR(correctInput,{packedCorrectData.offsets[index]}),{index});")
   print("  return (InferenceOutput){};")
   print("}")

   try:
      os.makedirs(outputLocation)
   except:
      pass

   with open(os.path.join(outputLocation,"model.bin"),"wb") as f:
      f.write(packedInitializers.data)

   with open(os.path.join(outputLocation,"correctOutput.bin"),"wb") as f:
      f.write(packedCorrectData.data)

# TODO: Need to take care with alignment issues. Embedded usually cannot handle misaligned data.

# TODO: Need to start giving NAMESPACE names to C stuff, this code is supposed to be easy to integrate anywhere.

# TODO: Need to also output the input file so we can start testing things and implementing the operators in C code.
#       Need to output parameters. Do not know the approach that we should take here.
#         Should we generate optimized functions for a given set of parameters?
#         Or should we just embed the data in source code and let runtime deal with it?
#         Or should we just pack it into a file and load at runtime? (Should only be a few kbs).
#       Even if we embed the data, I think it would be helpful to make a Conv2D especial function.

