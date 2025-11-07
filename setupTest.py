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


class TestType(Enum):
    GENERATED = (auto(),)
    FIXED = auto()
    FIXED_LIST = auto()


@dataclass
class SubTest:
    name: str
    path: str


@dataclass
class Test:
    type: TestType
    subTest: list[SubTest] = None
    focusLayer: int | None = None


def ValidateTest(test: Test):
    if test.type == TestType.FIXED:
        assert test.path != None


def ParseTest(testName, testJsonContent):
    try:
        testType = TestType[testJsonContent["type"]]
        path = testJsonContent.get("path", None)
        focusLayer = testJsonContent.get("focusLayer", None)
        subTests = testJsonContent.get("subTests", None)

        parsedSubTests = [SubTest(testName, path)]
        if subTests:
            parsedSubTests = [SubTest(x["name"], x["path"]) for x in subTests]

        test = Test(testType, parsedSubTests, focusLayer)
        ValidateTest(test)

        return test
    except Exception as e:
        print(f"Error parsing the json: {e}")
        sys.exit(-1)


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

    parser = argparse.ArgumentParser(
        prog="Tester",
        description="Setup for a single test",
    )
    parser.add_argument("TestName", choices=testNames)

    arguments = parser.parse_args()

    testName = arguments.TestName
    test = ParseTest(testName, testInfoJson[testName])

    # NOTE: We run from the makefile since we need to enter the python environment but we do not want to run this script from inside the environment.
    # TODO: Can we run from inside the environment and call nix? We cannot do the inverse I think but I do not know if we tried nix from inside python env.

    createVCD = False
    if test.type == TestType.GENERATED:
        createVCD = True

    boolStr = "true" if createVCD else "false"
    with open("./software/src/testInfo.h", "w") as f:
        f.write("\n".join([f'#include "{x.name}_modelInfo.h"' for x in test.subTest]))
        f.write("\n\n")

        f.write(f'#define TEST_NAME "{testName}"\n')
        f.write(f"#define CREATE_VCD {boolStr}\n\n")

        f.write(f"static TestModelInfo* testModels[] = " + "{\n")
        f.write(",\n".join([f"  &{x.name}_ModelInfo" for x in test.subTest]))
        f.write("\n};\n")

    if test.type == TestType.GENERATED:
        GenerateSimpleTest("./tests/generated/")
        GenerateDebug(
            "tests/generated/", "model.onnx", "software/", "software/src", testName
        )
    elif test.type == TestType.FIXED:
        GenerateDebug(
            test.subTest[0].path,
            "model.onnx",
            "software/",
            "software/src",
            testName,
            test.focusLayer,
        )

    elif test.type == TestType.FIXED_LIST:
        for subTest in test.subTest:
            GenerateDebug(
                subTest.path,
                "model.onnx",
                "software/",
                "software/src",
                subTest.name,
                test.focusLayer,
            )
