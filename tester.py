from scripts.onnxTest import GenerateDebug

GenerateDebug("tests/mnist_v7","output/")

# TODO: First PR.
#       I do not want to store the tests inside github. Offer a simple script that downloads and unpacks the tests inside the repo
#       Add the packing of the inputs data into an array and the logic inside the tester.c to load that file and to setup the inputs array.
#       Generate a header file that contains the info needed by the tester.c. Remove the functions that return the amount of data, using defines is better because the compiler will known it is a constant.

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