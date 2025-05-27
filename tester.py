from scripts.onnxTest import GenerateDebug,GenerateModelFromOnnxModel
import onnx
import sys

from scripts.onnxOperators import *

def MakeHashable(value: any):
   if(isinstance(value,list)):
      return tuple(value)
   else:
      return value

def JoinAttributes(attributeList,attribute):
   for name,attr in attribute.items():
      attributeList[name].add(MakeHashable(attr.value))

   return attributeList

def MakeAttributeList(attribute):
   res = {}
   for name,attr in attribute.items():
      res[name] = set()
      res[name].add(MakeHashable(attr.value))
   return res

def ExtractModelOperatorsTestableInfo(cModel):
   opToAttributesTested = {}
   for op in cModel.operations:
      opName = op.opName
      attributes = GetAttributesForOperator(op)

      if not attributes:
         continue
      
      if(opName in opToAttributesTested):
         opToAttributesTested[opName] = JoinAttributes(opToAttributesTested[opName],attributes)
      else:
         opToAttributesTested[opName] = MakeAttributeList(attributes)

   return opToAttributesTested

def ReportOperatorsTestedByModel(opToAttributesTesedDict):
   for opName,attributesTested in opToAttributesTesedDict.items():
      print(opName)
      operatorSpec = operatorNameToSpec[opName]
      attributesSpec = operatorSpec.attributesDict

      for attrName,tested in attributesTested.items():
         spec = attributesSpec[attrName]

         differentTests = len(tested)
         testedContent = repr(tested)

         if(spec.attrType == OnnxAttributeType.INTEGER):
            print(f"  {attrName} is tested in {differentTests} different ways: {testedContent}")
         elif(spec.attrType == OnnxAttributeType.BOUNDED_INTEGER):
            allAllowedAmount = len(spec.allowedValues)
            if(differentTests == allAllowedAmount):
               print(f"  {attrName} is fully tested (possible values: {spec.allowedValues})")
            else:
               print(f"  {attrName} is tested in {differentTests} different ways: {testedContent} (missing {allAllowedAmount - differentTests} tests)")
         elif(spec.attrType == OnnxAttributeType.INTEGER_LIST):
            print(f"  {attrName} is tested in {differentTests} different ways: {testedContent}")
         elif(spec.attrType == OnnxAttributeType.BOUNDED_STRING):
            allAllowedAmount = len(spec.allowedValues)
            if(differentTests == allAllowedAmount):
               print(f"  {attrName} is fully tested (possible values: {spec.allowedValues})")
            else:
               print(f"  {attrName} is tested in {differentTests} different ways: {testedContent} (missing {allAllowedAmount - differentTests} tests)")
         else:
            assert(False)


report = ExtractModelOperatorsTestableInfo(GenerateModelFromOnnxModel(onnx.load(sys.argv[1])))

ReportOperatorsTestedByModel(report)

GenerateDebug(sys.argv[1],sys.argv[3],sys.argv[2])


# TODO: Some models contain variable sized inputs (and maybe outputs).
#       Before progressing further, need to find a small example that contains variable sized nodes. In fact, it would be best to check as many models as possible to see what is possible and wether we are missing some concept before diving too deep into code generation.

# Examples: Faster R-CNN.

#      The fact that we might have to support variable lengths means that we probably want to push as much of the logic to runtime as possible.
#      This also means that our memory trick to preallocate memory blocks does not work properly.
#        Technically, since the input size is the only thing that can change, we probably can always calculate everything given an input size.
#        The problem is the implementation that is based on linear programming, how to make it work properly.
#          Do we have another AddressGen situation where we need to output a bunch of ifs and elses to properly follow the correct path of memory allocations?

# TODO: I want to keep track of every combination of operators that are tested.
#       I want to run a tester and the tester will tell me if:
#         For operators with multiple padding schemes, which shcemes where tested or not.
#         For operators with broadcasting, which broadcasting types where applied.
#         For operators with multiple optional outputs, which amount was used or not.
#         For operators with integer parameters with no limit, which amount where used and their values.
#       Basically, I want to keep track of every combination that is worth testing and I want
#       to make it easy to see if for the given amount of models, wether we are missing some combination or not.
#       NOTE: Also, if possible, I would like to "inspect" a new test and the tester reports wether this new test would add any checks or not.
#             This should be easy if we can first "extract" a "report" from a test, and we can join reports together.
#             Afterwards, we just have to extract a report from a test, join with the rest of the report, and calculate the difference.

# TODO: Most model examples that we obtained contain only a single test. We could get a bunch of pictures and generate more testcases by changing the inputs.

