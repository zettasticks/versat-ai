#include "stdint.h"
#include "versat_ai.h"
int numberLayers = 1;
LayerInfo layers[] = {{"", "Add", 32}};
int64_t temp_0[] = {4, 2};
int64_t temp_1[] = {4, 2};
int64_t temp_2[] = {4, 2};
AddInfo AddInfos[1] = {{2, temp_0, temp_1, temp_2}};

InferenceOutput DebugRunInference(void *outputMemory, void *temporaryMemory,
                                  void **inputs, void *modelMemory,
                                  void *correctInput) {
  void *res_0 = Versat_Add(inputs[0], inputs[1], OFFSET_PTR(outputMemory, 0), 0,
                           &AddInfos[0]);
  AssertAlmostEqual(res_0, OFFSET_PTR(correctInput, 0), 0);
  return (InferenceOutput){};
}
