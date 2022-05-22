solver.exe: wave.cu
	nvcc wave.cu -o solver.exe -I../../include -L../../lib64 -lnetcdf -lio
clean:	
	rm solver.exe