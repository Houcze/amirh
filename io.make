modules/libio.so: src/io.cpp
	g++ -fPIC -c src/io.cpp -I./include -lnetcdf
	g++ -shared -o modules/libio.so io.o -lnetcdf
	rm io.o