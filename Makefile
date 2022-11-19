func.o: ./core/function/func.cu
	nvcc -c ./core/function/func.cu -I./core/function
all: func.o amirh.cu
	nvcc -c amirh.cu -I./
	nvcc -o amirh.exe amirh.o func.o -lnetcdf
clean:
	rm *.o *.exe