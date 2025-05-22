from scripts.onnxTest import GenerateDebug
import sys

GenerateDebug(sys.argv[1],sys.argv[2])

# TODO: Some models contain variable sized inputs (and maybe outputs).
#       Before progressing further, need to find a small example that contains variable sized nodes. In fact, it would be best to check as many models as possible to see what is possible and wether we are missing some concept before diving too deep into code generation.

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

