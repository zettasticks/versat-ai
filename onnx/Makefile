
test:
	rm -rf output
	mkdir output
	python3 ./tester.py tests/SimpleAdd/model.onnx output tests/SimpleAdd
#	gcc -Wall -g -o output/exe tester.c software/staticSource.c output/code.c -Isoftware
#	cd output; ./exe

test-info:
	python3 ./tester.py tests/mnist_v7/model.onnx

clean:
	rm -rf output

full-clean: clean
	rm -rf tests python