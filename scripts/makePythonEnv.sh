#!/bin/bash

mkdir ../../python_env
pushd ../../python_env
python3 -m "venv" .
source ./bin/activate
pip install onnx==1.17.0 skl2onnx==1.17.0 matplotlib==3.10.3 onnxruntime==1.22.0 pydot==4.0.1 pulp==3.2.1
