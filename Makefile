build/amirh.exe: src/* modules/*
	nvcc src/amirh.cu -o build/amirh.exe -I./include -L./modules -lnetcdf -lio
clean:	
	rm build/amirh.exe
run:
	./build/amirh.exe ./build/simple_xy.nc data