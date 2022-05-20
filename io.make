lib64/libio.so: src/io.cpp
	g++ -fPIC -c src/io.cpp -I./include -lnetcdf
	g++ -shared -o lib64/libio.so io.o -lnetcdf
	rm io.o