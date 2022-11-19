func.o: ./core/function/func.cu
	nvcc -g -c ./core/function/func.cu -I./core/function
all: func.o amirh.cu
	nvcc -g -c amirh.cu -I./
	nvcc -g -o amirh.exe amirh.o func.o -lnetcdf
clean:
	rm *.o *.exe