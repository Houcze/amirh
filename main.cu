#include <matrix.cpp>
#include <iostream>
#include <cstdlib>
#include <vector>


int main(int argc, char* argv[])
{
	array a{argv[1], argv[2]};
	
	array b = a - a * 2. + a * 3.;
	long long int size;
	size = b.get_size();
	
	
	double *data;
	data = (double *) std::malloc(size * sizeof(double));
	int status = b.dataSync(data);

	auto shape = a.get_shape();
	
	
	for(int i=0; i<a.get_size(); i++)
	{
		std::cout << *(data + i) << '\t';
	}
			
}
