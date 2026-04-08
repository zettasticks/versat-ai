#!/bin/bash

majorV="$(python -c 'import sys; print(".".join(map(str, sys.version_info[0:1])))')"
minorV="$(python -c 'import sys; print(".".join(map(str, sys.version_info[1:2])))')"

echo "$majorV"
echo "$minorV"

if [[ $majorV != 3 ]]; then
   echo "Error, we require Python3"
else
   mkdir ../python_env
   pushd ../python_env
   python3 -m "venv" .
   source ./bin/activate

   if [[ $minorV -ge 11 ]]; then
      pip install onnx==1.17.0 skl2onnx==1.17.0 matplotlib==3.10.3 onnxruntime==1.24.4 pydot==4.0.1 pulp==3.2.1
   else
      pip install onnx==1.17.0 skl2onnx==1.17.0 matplotlib==3.10.3 onnxruntime==1.22.0 pydot==4.0.1 pulp==3.2.1
   fi
fi

