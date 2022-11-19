/*
 * 禁止一切在逻辑中直接调用核函数的行为
 * 这个文件里的函数主要实现了各种运算，编程范式为
 * func_k 为核函数，使用__global__修饰符
 * func 为host函数，不使用修饰符
 * 
*/

#include <cmath>
#include <func.h>
#include <vector>
#include <cuda_runtime.h>
static const int threadsperblock = 1024;


__global__ void add_k(double* input, double1 coeff, double* result, int1 d)
{
    /*
    * 矩阵+常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x)
    {
        result[index] = input[index] - coeff.x;
    }
}


__global__ void add_k(double* input, double1 coeff, double* result, int2 d)
{
    /*
    * 矩阵+常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y)
    {
        result[index] = input[index] + coeff.x;
    }
}


__global__ void add_k(double* input, double1 coeff, double* result, int3 d)
{
    /*
    * 矩阵+常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y * d.z)
    {
        result[index] = input[index] + coeff.x;
    }
}


__global__ void sub_k(double* input, double1 coeff, double* result, int1 d)
{
    /*
    * 矩阵减去常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x)
    {
        result[index] = input[index] - coeff.x;
    }
}


__global__ void sub_k(double* input, double1 coeff, double* result, int2 d)
{
    /*
    * 矩阵减去常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y)
    {
        result[index] = input[index] - coeff.x;
    }
}


__global__ void sub_k(double* input, double1 coeff, double* result, int3 d)
{
    /*
    * 矩阵减去常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y * d.z)
    {
        result[index] = input[index] - coeff.x;
    }
}


__global__ void mul_k(double* input, double1 coeff, double* result, int1 d)
{
    /*
    * 矩阵*常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x)
    {
        result[index] = input[index] * coeff.x;
    }
}


__global__ void mul_k(double* input, double1 coeff, double* result, int2 d)
{
    /*
    * 矩阵*常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y)
    {
        result[index] = input[index] * coeff.x;
    }
}


__global__ void mul_k(double* input, double1 coeff, double* result, int3 d)
{
    /*
    * 矩阵*常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y * d.z)
    {
        result[index] = input[index] * coeff.x;
    }
}


__global__ void div_k(double* input, double1 coeff, double* result, int1 d)
{
    /*
    * 矩阵/常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x)
    {
        result[index] = input[index] / coeff.x;
    }
}


__global__ void div_k(double* input, double1 coeff, double* result, int2 d)
{
    /*
    * 矩阵/常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y)
    {
        result[index] = input[index] / coeff.x;
    }
}


__global__ void div_k(double* input, double1 coeff, double* result, int3 d)
{
    /*
    * 矩阵/常数
    */
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y * d.z)
    {
        result[index] = input[index] / coeff.x;
    }
}


__global__ void add_k(double* x, double* y, double* result, int1 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x)
    {
        result[index] = x[index] + y[index];
    }    
       
}


__global__ void add_k(double* x, double* y, double* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y)
    {
        result[index] = x[index] + y[index];
    }    
       
}


__global__ void add_k(double* x, double* y, double* result, int3 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y * d.z)
    {
        result[index] = x[index] + y[index];
    }    
       
}


__global__ void sub_k(double* x, double* y, double* result, int1 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x)
    {
        result[index] = x[index] - y[index];
    }    
       
}


__global__ void sub_k(double* x, double* y, double* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y)
    {
        result[index] = x[index] - y[index];
    }    
       
}


__global__ void sub_k(double* x, double* y, double* result, int3 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y * d.z)
    {
        result[index] = x[index] - y[index];
    }    
       
}


__global__ void mul_k(double* x, double* y, double* result, int1 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x)
    {
        result[index] = x[index] * y[index];
    }    
       
}


__global__ void mul_k(double* x, double* y, double* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y)
    {
        result[index] = x[index] * y[index];
    }    
       
}


__global__ void mul_k(double* x, double* y, double* result, int3 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y * d.z)
    {
        result[index] = x[index] / y[index];
    }    
       
}


__global__ void div_k(double* x, double* y, double* result, int1 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x)
    {
        result[index] = x[index] / y[index];
    }    
       
}


__global__ void div_k(double* x, double* y, double* result, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y)
    {
        result[index] = x[index] / y[index];
    }    
       
}


__global__ void div_k(double* x, double* y, double* result, int3 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y * d.z)
    {
        result[index] = x[index] / y[index];
    }    
       
}


int add(double* x, double* y, double* result, int N1)
{
    add_k<<<ceil(double(N1) / threadsperblock), threadsperblock>>>(x, y, result, make_int1(N1));
    return EXIT_SUCCESS;
}


int add(double* x, double* y, double* result, int N1, int N2)
{
    add_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(x, y, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


int add(double* x, double* y, double* result, int N1, int N2, int N3)
{
    add_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(x, y, result, make_int3(N1, N2, N3));
    return EXIT_SUCCESS;
}


int sub(double* x, double* y, double* result, int N1)
{
    sub_k<<<ceil(double(N1) / threadsperblock), threadsperblock>>>(x, y, result, make_int1(N1));
    return EXIT_SUCCESS;
}


int sub(double* x, double* y, double* result, int N1, int N2)
{
    sub_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(x, y, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


int sub(double* x, double* y, double* result, int N1, int N2, int N3)
{
    sub_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(x, y, result, make_int3(N1, N2, N3));
    return EXIT_SUCCESS;
}


int mul(double* x, double* y, double* result, int N1)
{
    mul_k<<<ceil(double(N1) / threadsperblock), threadsperblock>>>(x, y, result, make_int1(N1));
    return EXIT_SUCCESS;
}


int mul(double* x, double* y, double* result, int N1, int N2)
{
    mul_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(x, y, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


int mul(double* x, double* y, double* result, int N1, int N2, int N3)
{
    mul_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(x, y, result, make_int3(N1, N2, N3));
    return EXIT_SUCCESS;
}


int div(double* x, double* y, double* result, int N1)
{
    div_k<<<ceil(double(N1) / threadsperblock), threadsperblock>>>(x, y, result, make_int1(N1));
    return EXIT_SUCCESS;
}


int div(double* x, double* y, double* result, int N1, int N2)
{
    div_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(x, y, result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


int div(double* x, double* y, double* result, int N1, int N2, int N3)
{
    div_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(x, y, result, make_int3(N1, N2, N3));
    return EXIT_SUCCESS;
}


/*
* 以下函数实现了矩阵与常数的运算，节省开销版本
*/
int add(double* x, double y, double* result, int N1)
{
    add_k<<<ceil(double(N1) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int1(N1));
    return EXIT_SUCCESS;
}


int add(double* x, double y, double* result, int N1, int N2)
{
    add_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


int add(double* x, double y, double* result, int N1, int N2, int N3)
{
    add_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int3(N1, N2, N3));
    return EXIT_SUCCESS;
}


int sub(double* x, double y, double* result, int N1)
{
    sub_k<<<ceil(double(N1) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int1(N1));
    return EXIT_SUCCESS;
}


int sub(double* x, double y, double* result, int N1, int N2)
{
    sub_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


int sub(double* x, double y, double* result, int N1, int N2, int N3)
{
    sub_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int3(N1, N2, N3));
    return EXIT_SUCCESS;
}


int mul(double* x, double y, double* result, int N1)
{
    mul_k<<<ceil(double(N1) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int1(N1));
    return EXIT_SUCCESS;
}


int mul(double* x, double y, double* result, int N1, int N2)
{
    mul_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


int mul(double* x, double y, double* result, int N1, int N2, int N3)
{
    mul_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int3(N1, N2, N3));
    return EXIT_SUCCESS;
}


int div(double* x, double y, double* result, int N1)
{
    div_k<<<ceil(double(N1) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int1(N1));
    return EXIT_SUCCESS;
}


int div(double* x, double y, double* result, int N1, int N2)
{
    div_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int2(N1, N2));
    return EXIT_SUCCESS;
}


int div(double* x, double y, double* result, int N1, int N2, int N3)
{
    div_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(x, make_double1(y), result, make_int3(N1, N2, N3));
    return EXIT_SUCCESS;
}
/*------------------------------------------------------------------------------------------------------*/


__global__ void expectation_k(double* x, double* y, double* result, double2 w, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y)
    {
        result[index] = (w.x) * x[index] + (w.y) * y[index];
    }
}


__global__ void expectation_k(double* x, double* y, double* z, double* result, double3 w, int3 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    
    if(index < d.x * d.y)
    {
        result[index] = (w.x) * x[index] + (w.y) * y[index] + (w.z) * z[index];
    }
}



__global__ void parti_diff_k(double* phi, double* result, int2 d, int2 ij)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = x_index;
    
    int i = ij.x;
    int j = ij.y;

    if(((index / d.y + i) < d.x) && ((index / d.y + i) >= 0) && ((index % d.y + j) < d.y) && ((index % d.y + j) >= 0))
    {
        result[index + i * d.y + j] = phi[index];
    }
}


int parti_diff(double* phi, double* result, int N1, int N2, int i, int j)
{
    parti_diff_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(i, j));
    return EXIT_SUCCESS;
}


__global__ void parti_diff_add_k(double* phi, double* result, int2 d, int2 ij)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = x_index;
    
    int i = ij.x;
    int j = ij.y;

    if(((index / d.y + i) < d.x) && ((index / d.y + i) >= 0) && ((index % d.y + j) < d.y) && ((index % d.y + j) >= 0))
    {
        result[index + i * d.y + j] += phi[index];
    }
}


int parti_diff_add(double* phi, double* result, int N1, int N2, int i, int j)
{
    parti_diff_add_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(i, j));
    return EXIT_SUCCESS;
}



__global__ void parti_diff_sub_k(double* phi, double* result, int2 d, int2 ij)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int index = x_index;
    
    int i = ij.x;
    int j = ij.y;

    if(((index / d.y + i) < d.x) && ((index / d.y + i) >= 0) && ((index % d.y + j) < d.y) && ((index % d.y + j) >= 0))
    {
        result[index + i * d.y + j] -= phi[index];
    }
}


int parti_diff_sub(double* phi, double* result, int N1, int N2, int i, int j)
{
    parti_diff_sub_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(i, j));
    return EXIT_SUCCESS;
}



int laplace(double* phi, double* result, int N1, int N2)
{
    cudaMemset(result, 0, N1 * N2 * sizeof(double));
    parti_diff_add_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(1, 0));
    parti_diff_add_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(-1, 0));
    parti_diff_add_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(0, 1));
    parti_diff_add_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(phi, result, make_int2(N1, N2), make_int2(0, -1));
    expectation_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(result, phi, result, make_double2(1, -4), make_int2(N1, N2));
    return EXIT_SUCCESS;
}



int expectation(double* x, double* y, double* result, int w1, int w2, int N1, int N2)
{
    expectation_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(x, y, result, make_double2(w1, w2), make_int2(N1, N2));
    return EXIT_SUCCESS;   
}


int expectation(double* x, double* y, double* z, double* result, int w1, int w2, int w3, int N1, int N2, int N3)
{
    expectation_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(x, y, z, result, make_double3(w1, w2, w3), make_int3(N1, N2, N3));
    return EXIT_SUCCESS;   
}


__global__ void sin_k(double* input, double* result, int1 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x)
    {
        result[index] = sin(input[index]);
    }    
       
}


__global__ void cos_k(double* input, double* result, int1 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x)
    {
        result[index] = cos(input[index]);
    }          
}


__global__ void tan_k(double* input, double* result, int1 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x)
    {
        result[index] = tan(input[index]);
    }          
}


int sin(double* input, double* result, int N1, int N2, int N3)
{
    sin_k<<<dim3(std::ceil(double(N1 * N2 * N3) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N1 * N2 * N3));
    return EXIT_SUCCESS;   
}


int cos(double* input, double* result, int N1, int N2, int N3)
{
    cos_k<<<dim3(std::ceil(double(N1 * N2 * N3) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N1 * N2 * N3));
    return EXIT_SUCCESS;   
}


int tan(double* input, double* result, int N1, int N2, int N3)
{
    tan_k<<<dim3(std::ceil(double(N1 * N2 * N3) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N1 * N2 * N3));
    return EXIT_SUCCESS;   
}


int sin(double* input, double* result, int N1, int N2)
{
    sin_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N1 * N2));
    return EXIT_SUCCESS;   
}


int cos(double* input, double* result, int N1, int N2)
{
    cos_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N1 * N2));
    return EXIT_SUCCESS;   
}


int tan(double* input, double* result, int N1, int N2)
{
    tan_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N1 * N2));
    return EXIT_SUCCESS;   
}


int sin(double* input, double* result, int N)
{
    sin_k<<<dim3(std::ceil(double(N) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N));
    return EXIT_SUCCESS;   
}


int cos(double* input, double* result, int N)
{
    cos_k<<<dim3(std::ceil(double(N) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N));
    return EXIT_SUCCESS;   
}


int tan(double* input, double* result, int N)
{
    tan_k<<<dim3(std::ceil(double(N) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(input, result, make_int1(N));
    return EXIT_SUCCESS;   
}


__global__ void set_cons_k(double* arr, double1 value, int1 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x)
    {
        arr[index] = value.x;
    }          
}



__global__ void set_cons_k(double* arr, double1 value, int2 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y)
    {
        arr[index] = value.x;
    }          
}



__global__ void set_cons_k(double* arr, double1 value, int3 d)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index = x_index;
    if(index < d.x * d.y * d.z)
    {
        arr[index] = value.x;
    }          
}


int ones(double* arr, int N)
{
    set_cons_k<<<dim3(std::ceil(double(N) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(1.), make_int1(N));
    return EXIT_SUCCESS;     
}


int ones(double* arr, int N1, int N2)
{
    set_cons_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(1.), make_int1(N1 * N2));
    return EXIT_SUCCESS;     
}


int ones(double* arr, int N1, int N2, int N3)
{
    set_cons_k<<<dim3(std::ceil(double(N1 * N2 * N3) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(1.), make_int1(N1 * N2 * N3));
    return EXIT_SUCCESS;     
}


int zeros(double* arr, int N)
{
    set_cons_k<<<dim3(std::ceil(double(N) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(0.), make_int1(N));
    return EXIT_SUCCESS;     
}


int zeros(double* arr, int N1, int N2)
{
    set_cons_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(0.), make_int1(N1 * N2));
    return EXIT_SUCCESS;     
}


int zeros(double* arr, int N1, int N2, int N3)
{
    set_cons_k<<<dim3(std::ceil(double(N1 * N2 * N3) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(0.), make_int1(N1 * N2 * N3));
    return EXIT_SUCCESS;     
}


int set_cons(double* arr, double value, int N)
{
    set_cons_k<<<dim3(std::ceil(double(N) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(value), make_int1(N));
    return EXIT_SUCCESS;     
}


int set_cons(double* arr, double value, int N1, int N2)
{
    set_cons_k<<<dim3(std::ceil(double(N1 * N2) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(value), make_int1(N1 * N2));
    return EXIT_SUCCESS;     
}


int set_cons(double* arr, double value, int N1, int N2, int N3)
{
    set_cons_k<<<dim3(std::ceil(double(N1 * N2 * N3) / threadsperblock), 1, 1), dim3(threadsperblock, 1, 1)>>>(arr, make_double1(value), make_int1(N1 * N2 * N3));
    return EXIT_SUCCESS;     
}



int get_gpuc()
{
    int gpuc;
    cudaGetDeviceCount(&gpuc);
    return gpuc;
}


int prod(std::vector<int> p)
{
    int size{1};
    for(const auto& it : p)
    {
        size *= it;
    }
    return size;
}


int seq_add(std::vector<double*> vlist, double* result, int N1, int N2, int N3)
{
    /*
     * 对三个维度的变量队列的操作
    */
    cudaMemset(result, 0, sizeof(double) * N1 * N2 * N3);
    
    for(auto& v : vlist)
    {
	    add_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(v, result, result, make_int3(N1, N2, N3));
    }
    return EXIT_SUCCESS;
}


int seq_mul(std::vector<double*> vlist, double* result, int N1, int N2, int N3)
{
    ones(result, N1, N2, N3);
    
    for(auto& v : vlist)
    {
	    mul_k<<<ceil(double(N1 * N2 * N3) / threadsperblock), threadsperblock>>>(v, result, result, make_int3(N1, N2, N3));
    }
    return EXIT_SUCCESS;
}


int seq_add(std::vector<double*> vlist, double* result, int N1, int N2)
{
    /*
     * 对2个维度的变量队列的操作
    */
    cudaMemset(result, 0, sizeof(double) * N1 * N2);
    
    for(auto& v : vlist)
    {
	    add_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(v, result, result, make_int2(N1, N2));
    }
    return EXIT_SUCCESS;
}


int seq_mul(std::vector<double*> vlist, double* result, int N1, int N2)
{
    ones(result, N1, N2);
    
    for(auto& v : vlist)
    {
	    mul_k<<<ceil(double(N1 * N2) / threadsperblock), threadsperblock>>>(v, result, result, make_int2(N1, N2));
    }
    return EXIT_SUCCESS;
}

/*
int Parti(double* input, double* result, int N1, int N2)
{
    ;
}


int Parti(double* input, double* result, int N1, int N2, int N3)
{
    ;
}
*/
