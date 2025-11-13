#!/usr/bin/python3

import sys
import json
import argparse
import subprocess as sp
from enum import Enum, auto
from dataclasses import dataclass

sys.path.append("./scripts")

from generateSimpleTests import GenerateSimpleTest
from onnxMain import GenerateDebug

# TODO: FIXED_LIST instead of encoding the models, it could encode the name of the tests themselves.
#       That way we would not need to repeat the path and focusLayer everytime, we just create a Fixed test and then
#       add the name of the test to the Fixed_list


class TestType(Enum):
    GENERATED = auto()
    FIXED = auto()
    FIXED_LIST = auto()


class TestMode(Enum):
    SOFTWARE = auto()
    VERSAT = auto()


@dataclass
class SubTest:
    name: str
    path: str
    focusLayer: int | None = None


@dataclass
class Test:
    type: TestType
    mode: TestMode
    subTest: list[SubTest] = None


def ValidateTest(test: Test):
    if test.type == TestType.FIXED:
        assert test.subTest[0].path != None


def ParseTest(testName, testInfo):
    try:
        splitted = testName.split("_")
        focusLayer = None

        if len(splitted) >= 2:
            focusLayer = int(splitted[-1])
            testName = "".join(splitted[:-1])

        testJsonContent = testInfo[testName]

        testType = TestType[testJsonContent["type"]]
        try:
            testMode = TestMode[testJsonContent["mode"]]
        except:
            testMode = TestMode.VERSAT

        path = testJsonContent.get("path", None)
        subTests = testJsonContent.get("subTests", None)

        parsedSubTests = [SubTest(testName, path, focusLayer)]
        if subTests:
            parsedSubTests = [
                SubTest(x["name"], x["path"], focusLayer) for x in subTests
            ]

        test = Test(testType, testMode, parsedSubTests)
        ValidateTest(test)

        return test
    except Exception as e:
        print(f"Error parsing the json: {e}")
        sys.exit(-1)


def SubTestName(subTest):
    if subTest.focusLayer:
        return subTest.name + "_" + str(subTest.focusLayer)
    else:
        return subTest.name


if __name__ == "__main__":
    testInfoJson = None
    jsonTestInfoPath = "tests.json"

    try:
        with open(jsonTestInfoPath, "r") as file:
            testInfoJson = json.load(file)
    except Exception as e:
        print(f"Error parsing json file that should contain tests info: {e}")
        sys.exit(-1)

    testNames = [x for x in testInfoJson.keys()]

    print(testNames)

    if len(sys.argv) != 2:
        print(
            "Need one and only one argument, the test name (append a final _<focusLayer> to only perform one layer of the test)"
        )
        sys.exit(-1)

    testName = sys.argv[1]
    test = ParseTest(testName, testInfoJson)

    # NOTE: We run from the makefile since we need to enter the python environment but we do not want to run this script from inside the environment.
    # TODO: Can we run from inside the environment and call nix? We cannot do the inverse I think but I do not know if we tried nix from inside python env.

    createVCD = False
    if test.type == TestType.GENERATED:
        createVCD = True

    boolStr = "true" if createVCD else "false"
    with open("./software/src/testInfo.h", "w") as f:
        f.write(
            "\n".join(
                [f'#include "{SubTestName(x)}_modelInfo.h"' for x in test.subTest]
            )
        )
        f.write("\n\n")

        f.write(f'#define TEST_NAME "{testName}"\n')
        f.write(f"#define CREATE_VCD {boolStr}\n\n")

        f.write(f"static TestModelInfo* testModels[] = " + "{\n")
        f.write(",\n".join([f"  &{SubTestName(x)}_ModelInfo" for x in test.subTest]))
        f.write("\n};\n")

    if test.type == TestType.GENERATED:
        GenerateSimpleTest("./tests/generated/")
        GenerateDebug(
            "tests/generated/",
            "model.onnx",
            "software/",
            "software/src",
            testName,
            None,
            test.mode == TestMode.SOFTWARE,
        )
    elif test.type == TestType.FIXED or test.type == TestType.FIXED_LIST:
        for subTest in test.subTest:
            GenerateDebug(
                subTest.path,
                "model.onnx",
                "software/",
                "software/src",
                SubTestName(subTest),
                subTest.focusLayer,
            )
