#include <tensor.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#define d 1
#define dx 1
#define dy 1
double omega = 0.0007292;


void Jpp(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / (4 * d * d)) * (
		(bias(*A, 0, 1) - bias(*A, 0, -1)) * (bias(*B, -1, 0) - bias(*B, 1, 0)) - 
		(bias(*A, -1, 0)  - bias(*A, 1, 0)) * (bias(*B, 0, 1) - bias(*B, 0, -1)) 
	);

}

void Jpm1(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / (4 * d * d)) * (
		bias(*A, -1, 1) * (bias(*B, -1, 0) - bias(*B, 0, 1)) -
		bias(*A, 1, -1) * (bias(*B, 0, -1) - bias(*B, 1, 0)) + 
		bias(*A, -1, -1) * (bias(*B, 0, -1) - bias(*B, -1, 0)) - 
		bias(*A, 1, 1) * (bias(*B, 1, 0) - bias(*B, 0, 1))
	);
}

void Jmp1(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / (4 * d * d)) * (
		bias(*A, 0, 1) * (bias(*B, -1, 1) - bias(*B, 1, 1)) - 
		bias(*A, 0, -1) * (bias(*B, -1, -1) - bias(*B, 1, -1)) - 
		bias(*A, -1, 0) * (bias(*B, -1, 1) - bias(*B, -1, -1)) + 
		bias(*A, 1, 0) * (bias(*B, 1, 1) - bias(*B, 1, -1))
	);
}

void Jmm(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / (8 * d * d)) * (
		(bias(*A, -1, 1) - bias(*A, 1, -1)) * (bias(*B, -1, -1) - bias(*B, 1, 1)) -
		(bias(*A, -1, -1) - bias(*A, 1, 1)) * (bias(*B, -1, 1) - bias(*B, 1, -1))
	);
}

void Jmp2(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1/ (8 * d * d)) * (
		bias(*A, -1, 1) * (bias(*B, -2, 0) - bias(*B, 0, 2)) - 
		bias(*A, 1, -1) * (bias(*B, 0, -2) - bias(*B, 0, 1)) - 
		bias(*A, -1, -1) * (bias(*B, -2, 0) - bias(*B, 0, -2)) + 
		bias(*A, 1, 1) * (bias(*B, 0, 2) - bias(*B, 2, 0)) 
	);
}

void Jpm2(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / ( 8 * d * d)) * (
        bias(*A, -2, 0) * (bias(*B, -1, -1) - bias(*B, -1, 1)) -
        bias(*A, 2, 0) * (bias(*B, 1, -1) - bias(*B, 1, 1)) -
        bias(*A, 0, 2) * (bias(*B, 1, -1) - bias(*B, -1, -1)) +
        bias(*A, 0, -2) * (bias(*B, 1, 1) - bias(*B, -1, 1))	
	);
}

void J1(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	io::cuda::tensor ja = *output;
	io::cuda::tensor jb = *output;
	io::cuda::tensor jc = *output;
	Jpp(A, B, &ja);
	Jpm1(A, B, &jb);
	Jmp1(A, B, &jc);
	*output = (1 / 3) * (ja + jb + jc);
}

void J2(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	io::cuda::tensor ja = *output;
	io::cuda::tensor jb = *output;
	io::cuda::tensor jc = *output;
	Jmm(A, B, &ja);
	Jmp2(A, B, &jb);
	Jpm2(A, B, &jc);
	*output = (1 / 3) * (ja + jb + jc);
}


void J(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	io::cuda::tensor j1 = *output;
	io::cuda::tensor j2 = *output;
	J1(A, B, &j1);
	J2(A, B, &j2);
	*output = 2 * j1 - j2;
}

void zeta(io::cuda::tensor *u, io::cuda::tensor *v, io::cuda::tensor *output)
{
	// \zeta = (\partial{v}) / (\partial{x}) - (\partial{u}) / (\partial{y})
	// in wind format
	io::cuda::tensor v_x = (3 * (*v) - 4 * bias(*v, -1, 0) + bias(*v, -2, 0)) / (2 * dy);
	io::cuda::tensor u_y = (3 * (*u) - 4 * bias(*u, 0, -1) + bias(*u, 0, -2)) / (2 * dx);
	*output = v_x - u_y;
}



int main(int argc, char* argv[])
{
	;
			
}
