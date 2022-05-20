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
		(io::cuda::bias(*A, 0, 1) - io::cuda::bias(*A, 0, -1)) * (io::cuda::bias(*B, -1, 0) - io::cuda::bias(*B, 1, 0)) - 
		(io::cuda::bias(*A, -1, 0)  - io::cuda::bias(*A, 1, 0)) * (io::cuda::bias(*B, 0, 1) - io::cuda::bias(*B, 0, -1)) 
	);

}

void Jpm1(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / (4 * d * d)) * (
		io::cuda::bias(*A, -1, 1) * (io::cuda::bias(*B, -1, 0) - io::cuda::bias(*B, 0, 1)) -
		io::cuda::bias(*A, 1, -1) * (io::cuda::bias(*B, 0, -1) - io::cuda::bias(*B, 1, 0)) + 
		io::cuda::bias(*A, -1, -1) * (io::cuda::bias(*B, 0, -1) - io::cuda::bias(*B, -1, 0)) - 
		io::cuda::bias(*A, 1, 1) * (io::cuda::bias(*B, 1, 0) - io::cuda::bias(*B, 0, 1))
	);
}

void Jmp1(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / (4 * d * d)) * (
		io::cuda::bias(*A, 0, 1) * (io::cuda::bias(*B, -1, 1) - io::cuda::bias(*B, 1, 1)) - 
		io::cuda::bias(*A, 0, -1) * (io::cuda::bias(*B, -1, -1) - io::cuda::bias(*B, 1, -1)) - 
		io::cuda::bias(*A, -1, 0) * (io::cuda::bias(*B, -1, 1) - io::cuda::bias(*B, -1, -1)) + 
		io::cuda::bias(*A, 1, 0) * (io::cuda::bias(*B, 1, 1) - io::cuda::bias(*B, 1, -1))
	);
}

void Jmm(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / (8 * d * d)) * (
		(io::cuda::bias(*A, -1, 1) - io::cuda::bias(*A, 1, -1)) * (io::cuda::bias(*B, -1, -1) - io::cuda::bias(*B, 1, 1)) -
		(io::cuda::bias(*A, -1, -1) - io::cuda::bias(*A, 1, 1)) * (io::cuda::bias(*B, -1, 1) - io::cuda::bias(*B, 1, -1))
	);
}

void Jmp2(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1/ (8 * d * d)) * (
		io::cuda::bias(*A, -1, 1) * (io::cuda::bias(*B, -2, 0) - io::cuda::bias(*B, 0, 2)) - 
		io::cuda::bias(*A, 1, -1) * (io::cuda::bias(*B, 0, -2) - io::cuda::bias(*B, 0, 1)) - 
		io::cuda::bias(*A, -1, -1) * (io::cuda::bias(*B, -2, 0) - io::cuda::bias(*B, 0, -2)) + 
		io::cuda::bias(*A, 1, 1) * (io::cuda::bias(*B, 0, 2) - io::cuda::bias(*B, 2, 0)) 
	);
}

void Jpm2(io::cuda::tensor *A, io::cuda::tensor *B, io::cuda::tensor *output)
{
	*output = (1 / ( 8 * d * d)) * (
        io::cuda::bias(*A, -2, 0) * (io::cuda::bias(*B, -1, -1) - io::cuda::bias(*B, -1, 1)) -
        io::cuda::bias(*A, 2, 0) * (io::cuda::bias(*B, 1, -1) - io::cuda::bias(*B, 1, 1)) -
        io::cuda::bias(*A, 0, 2) * (io::cuda::bias(*B, 1, -1) - io::cuda::bias(*B, -1, -1)) +
        io::cuda::bias(*A, 0, -2) * (io::cuda::bias(*B, 1, 1) - io::cuda::bias(*B, -1, 1))	
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
	io::cuda::tensor v_x = (3 * (*v) - 4 * io::cuda::bias(*v, -1, 0) + io::cuda::bias(*v, -2, 0)) / (2 * dy);
	io::cuda::tensor u_y = (3 * (*u) - 4 * io::cuda::bias(*u, 0, -1) + io::cuda::bias(*u, 0, -2)) / (2 * dx);
	*output = v_x - u_y;
}



int main(int argc, char* argv[])
{
	io::cpu::tensor a(argv[1], argv[2]);
	io::cpu::tensor b(argv[1], argv[2]);
	std::cout << "1" << a.data[0] << std::endl;
	std::cout << "2" << b.data[0] << std::endl;

	io::cuda::tensor c = io::cpu_to_cuda(a);
	io::cpu::tensor a2 = io::cuda_to_cpu(c);
	std::cout << "3" << a2.data[0] << std::endl;


	io::cuda::tensor e = io::cpu_to_cuda(b);
	io::cuda::tensor w = c + e;
	io::cpu::tensor w2 = io::cuda_to_cpu(w);
	std::cout << w2.data[0] << std::endl;
	std::cout << "No problem" << std::endl;
}
