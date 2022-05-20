build/amirh.exe: amirh.cu tensor.cu io.cpp
	nvcc src/amirh.cu -o build/amirh.exe -I./include -L./libs -lnetcdf -ltensor
clean:
	rm build/amirh.exe