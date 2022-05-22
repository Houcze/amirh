build/amirh.exe: src/* lib64/*
	nvcc src/amirh.cu -o build/amirh.exe -I./include -L./lib64 -lnetcdf -lio
clean:	
	rm build/amirh.exe
run:
	./build/amirh.exe ./build/simple_xy.nc data