#!/bin/bash

minorV="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[1:2])))')"

if [[ -z $minorV ]]; then
   echo "Error, we require python3 to be installed and accessible"
else
   mkdir ../python_env
   pushd ../python_env
   python3 -m "venv" .
   source ./bin/activate

   if [[ $minorV -ge 11 ]]; then
      pip install onnx==1.17.0 skl2onnx==1.17.0 matplotlib==3.10.3 onnxruntime==1.24.4 pydot==4.0.1
   else
      pip install onnx==1.17.0 skl2onnx==1.17.0 matplotlib==3.10.3 onnxruntime==1.22.0 pydot==4.0.1
   fi
fi
