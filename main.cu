#include <matrix.cpp>
#include <iostream>
#include <cstdlib>
#include <vector>
#define d 1

void Jpp(array *A, array *B, array *output)
{
	*output = (1 / (4 * d * d)) * (
		(bias(*A, 0, 1) - bias(*A, 0, -1)) * (bias(*B, -1, 0) - bias(*B, 1, 0)) - 
		(bias(*A, -1, 0)  - bias(*A, 1, 0)) * (bias(*B, 0, 1) - bias(*B, 0, -1)) 
	);
}

void Jpm1(array *A, array *B, array *output)
{
	*output = (1 / (4 * d * d)) * (
		bias(*A, -1, 1) * (bias(*B, -1, 0) - bias(*B, 0, 1)) -
		bias(*A, 1, -1) * (bias(*B, 0, -1) - bias(*B, 1, 0)) + 
		bias(*A, -1, -1) * (bias(*B, 0, -1) - bias(*B, -1, 0)) - 
		bias(*A, 1, 1) * (bias(*B, 1, 0) - bias(*B, 0, 1))
	);
}

void Jmp1(array *A, array *B, array *output)
{
	*output = (1 / (4 * d * d)) * (
		bias(*A, 0, 1) * (bias(*B, -1, 1) - bias(*B, 1, 1)) - 
		bias(*A, 0, -1) * (bias(*B, -1, -1) - bias(*B, 1, -1)) - 
		bias(*A, -1, 0) * (bias(*B, -1, 1) - bias(*B, -1, -1)) + 
		bias(*A, 1, 0) * (bias(*B, 1, 1) - bias(*B, 1, -1))
	);
}

void Jmm(array *A, array *B, array *output)
{
	*output = (1 / (8 * d * d)) * (
		(bias(*A, -1, 1) - bias(*A, 1, -1)) * (bias(*B, -1, -1) - bias(*B, 1, 1)) -
		(bias(*A, -1, -1) - bias(*A, 1, 1)) * (bias(*B, -1, 1) - bias(*B, 1, -1))
	);
}

void Jmp2(array *A, array *B, array *output)
{
	*output = (1/ (8 * d * d)) * (
		bias(*A, -1, 1) * (bias(*B, -2, 0) - bias(*B, 0, 2)) - 
		bias(*A, 1, -1) * (bias(*B, 0, -2) - bias(*B, 0, 1)) - 
		bias(*A, -1, -1) * (bias(*B, -2, 0) - bias(*B, 0, -2)) + 
		bias(*A, 1, 1) * (bias(*B, 0, 2) - bias(*B, 2, 0)) 
	);
}

void Jpm2(array *A, array *B, array *output)
{
	*output = (1 / ( 8 * d * d)) * (
        bias(*A, -2, 0) * (bias(*B, -1, -1) - bias(*B, -1, 1)) -
        bias(*A, 2, 0) * (bias(*B, 1, -1) - bias(*B, 1, 1)) -
        bias(*A, 0, 2) * (bias(*B, 1, -1) - bias(*B, -1, -1)) +
        bias(*A, 0, -2) * (bias(*B, 1, 1) - bias(*B, -1, 1))	
	);
}

void J1(array *A, array *B, array *output)
{
	array ja = *output;
	array jb = *output;
	array jc = *output;
	Jpp(A, B, &ja);
	Jpm1(A, B, &jb);
	Jmp1(A, B, &jc);
	*output = (1 / 3) * (ja + jb + jc);
}

void J2(array *A, array *B, array *output)
{
	array ja = *output;
	array jb = *output;
	array jc = *output;
	Jmm(A, B, &ja);
	Jmp2(A, B, &jb);
	Jpm2(A, B, &jc);
	*output = (1 / 3) * (ja + jb + jc);
}


void J(array *A, array *B, array *output)
{
	array j1 = *output;
	array j2 = *output;
	J1(A, B, &j1);
	J2(A, B, &j2);
	*output = 2 * j1 - j2;
}


int main(int argc, char* argv[])
{
	array a{argv[1], argv[2]};
	size_t size;
	size_t dims;
	size = a.get_size();
	dims = a.get_dims();
	size_t * shape;
	shape = (size_t *) std::malloc(dims * sizeof(size_t));
	shape = a.get_shape();
	array output{size, dims, shape};
	J(&a, &a, &output);

	double *data;
	data = (double *) std::malloc(size * sizeof(double));
	int status = output.dataSync(data);	
	for(int i=0; i<a.get_size(); i++)
	{
		std::cout << *(data + i) << '\t';
	}
			
}
