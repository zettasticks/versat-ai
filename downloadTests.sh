mkdir tests
wget --output-document ./tests/mnist_1.tar.gz https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-1.tar.gz
wget --output-document ./tests/mnist_7.tar.gz https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-7.tar.gz
wget --output-document ./tests/tiny_yolov2.tar.gz  https://github.com/onnx/models/raw/refs/heads/main/validated/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.tar.gz

pushd tests
tar xf mnist_1.tar.gz
mv mnist mnist_v1
tar xf mnist_7.tar.gz
mv model mnist_v7
tar xf tiny_yolov2.tar.gz
mv model tiny_yolov2_v7
popd