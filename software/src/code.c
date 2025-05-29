#include "stdint.h"
#include "versat_ai.h"
int numberLayers = 12;
LayerInfo layers[] = {{"Times212_reshape1", "Reshape", 10240},
                      {"Convolution28", "Conv", 25088},
                      {"Plus30", "Add", 25088},
                      {"ReLU32", "Relu", 25088},
                      {"Pooling66", "MaxPool", 6272},
                      {"Convolution110", "Conv", 12544},
                      {"Plus112", "Add", 12544},
                      {"ReLU114", "Relu", 12544},
                      {"Pooling160", "MaxPool", 1024},
                      {"Times212_reshape0", "Reshape", 1024},
                      {"Times212", "MatMul", 40},
                      {"Plus214", "Add", 40}};
int64_t temp_0[] = {16, 4, 4, 10};
int64_t temp_1[] = {1, 16, 4, 4};
int64_t temp_2[] = {1, 1, 28, 28};
int64_t temp_3[] = {8, 1, 5, 5};
int64_t temp_4[] = {1, 8, 28, 28};
int64_t temp_5[] = {1, 8, 14, 14};
int64_t temp_6[] = {16, 8, 5, 5};
int64_t temp_7[] = {1, 16, 14, 14};
int64_t temp_8[] = {1, 8, 28, 28};
int64_t temp_9[] = {1, 8, 1, 1};
int64_t temp_10[] = {1, 8, 28, 28};
int64_t temp_11[] = {1, 16, 14, 14};
int64_t temp_12[] = {1, 16, 1, 1};
int64_t temp_13[] = {1, 16, 14, 14};
int64_t temp_14[] = {1, 10};
int64_t temp_15[] = {1, 10};
int64_t temp_16[] = {1, 10};
int64_t temp_17[] = {1, 8, 28, 28};
int64_t temp_18[] = {1, 16, 14, 14};
int64_t temp_19[] = {1, 8, 28, 28};
int64_t temp_20[] = {1, 8, 14, 14};
int64_t temp_21[] = {1, 16, 14, 14};
int64_t temp_22[] = {1, 16, 4, 4};
int64_t temp_23[] = {1, 256};
int64_t temp_24[] = {256, 10};
int64_t temp_25[] = {1, 10};
ReshapeInfo ReshapeInfos[2] = {{temp_0, 4, 2}, {temp_1, 4, 2}};
ConvInfo ConvInfos[2] = {{4, temp_2, 4, temp_3, 4, temp_4},
                         {4, temp_5, 4, temp_6, 4, temp_7}};
AddInfo AddInfos[3] = {{4, temp_8, temp_9, temp_10},
                       {4, temp_11, temp_12, temp_13},
                       {2, temp_14, temp_15, temp_16}};
ReluInfo ReluInfos[2] = {{4, temp_17}, {4, temp_18}};
MaxPoolInfo MaxPoolInfos[2] = {{4, temp_19, temp_20, 2, 2},
                               {4, temp_21, temp_22, 3, 3}};
MatMulInfo MatMulInfos[1] = {{temp_23, 2, temp_24, 2, temp_25, 2}};

InferenceOutput DebugRunInference(void *outputMemory, void *temporaryMemory,
                                  void **inputs, void *modelMemory,
                                  void *correctInput) {
  void *res_0 =
      Reshape(OFFSET_PTR(modelMemory, 0), OFFSET_PTR(modelMemory, 10240),
              OFFSET_PTR(temporaryMemory, 25088), 0, &ReshapeInfos[0]);
  AssertAlmostEqual(res_0, OFFSET_PTR(correctInput, 0), 0);
  void *res_1 = Conv(inputs[0], OFFSET_PTR(modelMemory, 10256),
                     OFFSET_PTR(temporaryMemory, 35328), 1, &ConvInfos[0]);
  AssertAlmostEqual(res_1, OFFSET_PTR(correctInput, 10240), 1);
  void *res_2 =
      Add(OFFSET_PTR(correctInput, 10240), OFFSET_PTR(modelMemory, 11056),
          OFFSET_PTR(temporaryMemory, 0), 2, &AddInfos[0]);
  AssertAlmostEqual(res_2, OFFSET_PTR(correctInput, 35328), 2);
  void *res_3 = Relu(OFFSET_PTR(correctInput, 35328),
                     OFFSET_PTR(temporaryMemory, 35328), 3, &ReluInfos[0]);
  AssertAlmostEqual(res_3, OFFSET_PTR(correctInput, 60416), 3);
  void *res_4 = MaxPool(OFFSET_PTR(correctInput, 60416),
                        OFFSET_PTR(temporaryMemory, 0), 4, &MaxPoolInfos[0]);
  AssertAlmostEqual(res_4, OFFSET_PTR(correctInput, 85504), 4);
  void *res_5 =
      Conv(OFFSET_PTR(correctInput, 85504), OFFSET_PTR(modelMemory, 11088),
           OFFSET_PTR(temporaryMemory, 35328), 5, &ConvInfos[1]);
  AssertAlmostEqual(res_5, OFFSET_PTR(correctInput, 91776), 5);
  void *res_6 =
      Add(OFFSET_PTR(correctInput, 91776), OFFSET_PTR(modelMemory, 23888),
          OFFSET_PTR(temporaryMemory, 47872), 6, &AddInfos[1]);
  AssertAlmostEqual(res_6, OFFSET_PTR(correctInput, 104320), 6);
  void *res_7 = Relu(OFFSET_PTR(correctInput, 104320),
                     OFFSET_PTR(temporaryMemory, 0), 7, &ReluInfos[1]);
  AssertAlmostEqual(res_7, OFFSET_PTR(correctInput, 116864), 7);
  void *res_8 =
      MaxPool(OFFSET_PTR(correctInput, 116864),
              OFFSET_PTR(temporaryMemory, 24064), 8, &MaxPoolInfos[1]);
  AssertAlmostEqual(res_8, OFFSET_PTR(correctInput, 129408), 8);
  void *res_9 =
      Reshape(OFFSET_PTR(correctInput, 129408), OFFSET_PTR(modelMemory, 23952),
              OFFSET_PTR(temporaryMemory, 23040), 9, &ReshapeInfos[1]);
  AssertAlmostEqual(res_9, OFFSET_PTR(correctInput, 130432), 9);
  void *res_10 =
      MatMul(OFFSET_PTR(correctInput, 130432), OFFSET_PTR(correctInput, 0),
             OFFSET_PTR(temporaryMemory, 0), 10, &MatMulInfos[0]);
  AssertAlmostEqual(res_10, OFFSET_PTR(correctInput, 131456), 10);
  void *res_11 =
      Add(OFFSET_PTR(correctInput, 131456), OFFSET_PTR(modelMemory, 23968),
          OFFSET_PTR(outputMemory, 0), 11, &AddInfos[2]);
  AssertAlmostEqual(res_11, OFFSET_PTR(correctInput, 131496), 11);
  return (InferenceOutput){};
}
