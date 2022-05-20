libs/libtensor.a: src/tensor.cu
	nvcc -c src/tensor.cu -I./include -lnetcdf
	ar -crv libs/libtensor.a tensor.o
	rm tensor.o