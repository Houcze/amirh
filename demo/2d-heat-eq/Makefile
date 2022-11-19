func.o: ./core/function/func.cu
	nvcc -c ./core/function/func.cu -I./core/function
all: func.o 2d-heat-equation.cu
	nvcc -c 2d-heat-equation.cu -I./
	nvcc -o amirh.exe 2d-heat-equation.o func.o -lnetcdf
clean:
	rm *.o *.exe