func.o: ../core/function/func.cu
	nvcc -c ../core/function/func.cu -I../core/function -std=c++17
all: func.o t1.cu
	nvcc -c t1.cu -I./ -std=c++17
	nvcc -o t1.exe t1.o func.o -lnetcdf -std=c++17
clean:
	rm *.o *.exe