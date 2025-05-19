pushd python
python3 -m "venv" .
source ./bin/activate
pip install onnx==1.17.0 skl2onnx==1.17.0 matplotlib onnxruntime pydot pulp
