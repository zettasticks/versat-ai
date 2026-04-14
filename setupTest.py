#!/usr/bin/python3

import os
import sys
import json
import argparse
import subprocess as sp
import copy
from pprint import pprint
from enum import Enum, auto
from dataclasses import dataclass

sys.path.append("./scripts")

from generateSimpleTests import (
    GenerateTest,
    GenerateLite,
    GenerateHeavy,
    GenerateSoftmax,
)
from onnxMain import GenerateDebug

# TODO: FIXED_LIST instead of encoding the models, it could encode the name of the tests themselves.
#       That way we would not need to repeat the path and focusLayer everytime, we just create a Fixed test and then
#       add the name of the test to the Fixed_list


class TestType(Enum):
    GENERATED = auto()
    FIXED = auto()
    FIXED_LIST = auto()


class TestMode(Enum):
    DEFAULT = auto()
    SOFTWARE = auto()
    VERSAT = auto()


class GenType(Enum):
    NORMAL = auto()
    HEAVY = auto()
    LITE = auto()
    SOFTMAX = auto()


def OverrideTestMode(stronger, weaker):
    if stronger == TestMode.DEFAULT:
        return weaker

    return stronger


@dataclass
class TestConfiguraton:
    focusLayerRange: [int, int] = None
    mode: TestMode = TestMode.DEFAULT


@dataclass
class SubTest:
    name: str
    config: TestConfiguraton


@dataclass
class Test:
    type: TestType
    path: str
    subTest: list[SubTest] = None
    genType: GenType = None


def ParseTestName(testName):
    splitted = testName.split("_")

    originalName = splitted[0]

    config = TestConfiguraton()
    focusLayerRange = [-1, -1]
    seenOneInt = False
    for x in splitted[1:]:
        if x == "PC":
            config.mode = TestMode.SOFTWARE
        elif x == "VERSAT":
            config.mode = TestMode.VERSAT

        try:
            asInt = int(x)
            config.focusLayer = asInt

            if not seenOneInt:
                focusLayerRange[0] = asInt
                focusLayerRange[1] = asInt
                seenOneInt = True
            else:
                focusLayerRange[1] = asInt
        except:
            pass

    if seenOneInt:
        config.focusLayerRange = focusLayerRange

    return originalName, config


def ParseTest(testName, testInfo, allTests):
    testType = TestType[testInfo["type"]]
    path = testInfo.get("path", None)
    subTests = testInfo.get("subTests", None)

    genType = None
    try:
        genType = GenType[testInfo.get("genType", None)]
    except:
        pass

    parsedSubTests = [SubTest(testName, TestConfiguraton())]
    if subTests:
        parsedSubTests = []
        for name in subTests:
            subTestName, subTestConfigs = ParseTestName(name)
            sub = SubTest(subTestName, subTestConfigs)
            parsedSubTests.append(sub)

    test = Test(testType, path, parsedSubTests, genType)

    return test


def ParseTests(testInfoJson):
    tests = {}
    for name in testInfoJson:
        test = ParseTest(name, testInfoJson[name], tests)
        tests[name] = test

    return tests


def SetupTest(test, subTest):
    if test.type == TestType.GENERATED:
        if test.genType == GenType.NORMAL:
            GenerateTest(test.path)
        elif test.genType == GenType.LITE:
            GenerateLite(test.path)
        elif test.genType == GenType.HEAVY:
            GenerateHeavy(test.path)
        elif test.genType == GenType.SOFTMAX:
            GenerateSoftmax(test.path)
    else:
        assert test.type != TestType.FIXED_LIST

    GenerateDebug(
        test.path,
        "model.onnx",
        "resources/",
        "software/src",
        subTest.name,
        subTest.config.focusLayerRange,
        subTest.config.mode == TestMode.SOFTWARE,
    )


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
            "Need one and only one argument, the test name (append a final _<focusLayer> to only perform one layer of the test or _<focusStart>_<focusEnd> to perform N layers)"
        )
        sys.exit(-1)

    allTests = ParseTests(testInfoJson)
    properName, configs = ParseTestName(sys.argv[1])

    test = allTests[properName]

    for subTest in test.subTest:
        subTest.config.mode = OverrideTestMode(subTest.config.mode, configs.mode)
        if configs.focusLayerRange:
            subTest.config.focusLayerRange = configs.focusLayerRange

    # NOTE: We run from the makefile since we need to enter the python environment but we do not want to run this script from inside the environment.
    # TODO: Can we run from inside the environment and call nix? We cannot do the inverse I think but I do not know if we tried nix from inside python env.

    createVCD = False
    if test.type == TestType.GENERATED:
        createVCD = True

    os.makedirs("resources", exist_ok=True)

    boolStr = "true" if createVCD else "false"
    if False:
        with open("./resources/testInfo.h", "w") as f:
            f.write(
                "\n".join([f'#include "{x.name}_modelInfo.h"' for x in test.subTest])
            )
            f.write("\n\n")

            f.write(f'#define TEST_NAME "{properName}"\n')
            f.write(f"#define CREATE_VCD {boolStr}\n\n")

            f.write(f"static TestModelInfo* testModels[] = " + "{\n")
            f.write(",\n".join([f"  &{x.name}_ModelInfo" for x in test.subTest]))
            f.write("\n};\n")
    else:
        with open("./resources/VERSAT_TEST_METADATA.txt", "w") as f:
            for subTest in test.subTest:
                f.write(subTest.name + "\n")

    if test.type == TestType.FIXED_LIST:
        for subTest in test.subTest:
            properTest = allTests[subTest.name]
            SetupTest(properTest, subTest)
    else:
        SetupTest(test, test.subTest[0])
