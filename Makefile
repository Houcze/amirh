build/amirh.exe: src/* lib64/*
	nvcc src/amirh.cu -o build/amirh.exe -I./include -L./lib64 -lnetcdf -lio
clean:	
	rm build/amirh.exe

test: src/* lib64/*
	g++ test.cpp -o test.exe -I./include -L./lib64 -lnetcdf -lio
testclean:
	rm test.exe