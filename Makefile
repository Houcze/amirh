netcdf.o: ./io/netcdf.cpp
	g++ -fPIE -g -c ./io/netcdf.cpp -I./ -lnetcdf -std=c++17
Prop.o: ./core/function/Prop.cu
	nvcc -g -c ./core/function/Prop.cu -I./ -std=c++17
func.o: ./core/function/func.cu
	nvcc -g -c ./core/function/func.cu -I./core/function -std=c++17
mem.o: ./core/mem.cu
	nvcc -g -c ./core/mem.cu -I./ -std=c++17
Variables.o: ./core/variables/Variables.cu netcdf.o
	nvcc -g -c ./core/variables/Variables.cu netcdf.o -I./ -lnetcdf -std=c++17
all: func.o Prop.o mem.o Variables.o amirh.cu netcdf.o
	nvcc -g -c amirh.cu -I./ -std=c++17
	nvcc -o amirh.exe func.o Prop.o mem.o Variables.o amirh.o netcdf.o -lnetcdf -std=c++17
clean:
	rm *.o *.exe
