from dataclasses import dataclass
from copy import copy

from scripts.versatDefs import Operation

# TODO: It might be useful for us to generate the C structs from Python that to keep trying to match static C structs with the Python code that generates it.
#       The problem is that any Python change also causes us to have to change the C code, meaning that we do not save any trouble from this change.
#       If we eventually generate all the C code from Python, that this is something that we could do easily, since Python generates both code and data.
#       But this might be overkill. It is also easier to start this way and move runtime to python as we progress.

def ExtendShape(shapeList,dimensions):
   res = copy(shapeList)
   if(len(shapeList) < dimensions):
      for i in range(dimensions - len(shapeList)):
         res = [1] + res
   return res

def BroadCastShape(op0,op1):
   if(len(op0) != len(op1)):
      length = max(len(op0),len(op1))
      op0 = ExtendShape(op0)
      op1 = ExtendShape(op1)

   res = []
   for a,b in zip(op0,op1):
      res.append(max(a,b))
   return res


@dataclass
class OnnxOperatorSpec:
   name: str
   emitFunction: any

def EmitAdd(emitter,op : Operation):
   maxDims = max(len(op.inputDimensions[0]),len(op.inputDimensions[1]))

   op0 = ExtendShape(op.inputDimensions[0],maxDims)
   op1 = ExtendShape(op.inputDimensions[1],maxDims)

   broadCastedShape = BroadCastShape(op0,op1)

   aux_0 = emitter.EmitArray("int",op0)
   aux_1 = emitter.EmitArray("int",op1)
   aux_2 = emitter.EmitArray("int",broadCastedShape)

   return [maxDims,aux_0,aux_1,aux_2]

def EmitRelu(emitter,op : Operation):
   aux = emitter.EmitArray("int",op.inputDimensions[0])
   dims = len(op.inputDimensions[0])
   return [dims,aux]

def EmitMaxPool(emitter,op : Operation):
   dims = len(op.inputDimensions[0])
   inputShape = emitter.EmitArray("int",op.inputDimensions[0])
   outputShape = emitter.EmitArray("int",op.outputDimensions)

   kernel_shape = op.attributes['kernel_shape']

   return [dims,inputShape,outputShape,kernel_shape[0],kernel_shape[1]]

def IsOperatorRegistered(opName : str):
   return (opName in operatorNameToSpec)

def EmitParameterList(emitter,op : Operation):
   global operatorNameToSpec
   spec = operatorNameToSpec.get(op.opName,None)

   if(not spec):
      print(f"Operator {op.opName} is not registered and no implementation exists for it")
      print(f"Know operators: {operatorNameToSpec.keys()}")
   else:
      return spec.emitFunction(emitter,op)

# Register new operators here
operatorNameToSpec = {}
operatorNameToSpec['Add'] = OnnxOperatorSpec("Add",EmitAdd)
operatorNameToSpec['Relu'] = OnnxOperatorSpec("Relu",EmitRelu)
operatorNameToSpec['MaxPool'] = OnnxOperatorSpec("MaxPool",EmitMaxPool)