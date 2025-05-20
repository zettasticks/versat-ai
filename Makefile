
all:
	gcc -g -o exe tester.c software/staticSource.c
	./exe

test:
	rm -rf output
	mkdir output
	python3 ./tester.py tests/mnist_v7 output > output/code.c
	# -fsanitize=address -static-libasan
	gcc -Wall -g -o output/exe tester.c software/staticSource.c output/code.c -Isoftware
	./output/exe
	#gdb -ex run --args ./output/exe
