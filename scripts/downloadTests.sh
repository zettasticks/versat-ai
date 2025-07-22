#!/bin/bash

mkdir ../tests
wget --output-document ./tests/mnist_7.tar.gz https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-7.tar.gz

pushd ../tests
tar xf mnist_7.tar.gz
mv model mnist_v7
popd