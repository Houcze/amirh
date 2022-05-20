build/main.exe: main.cu tensor.cu io.cpp
	nvcc src/main.cu -o build/main.exe -lnetcdf
clean:
	rm build/main.exe