#include <cuda_runtime.h>
#include <functional>
#include <variant>
#include <iostream>
#include <../core/function/func.h>
#include <map>
#include <any>
#include <iomanip>
#include <type_traits>

#define NAME(variable) (#variable)
/*
 * Ops 类的作用是建立 Variable类和以double* 等基础类型为接口的函数之间的桥梁
 */

/* 矩阵之间四则运算的参数列表
 * F1 : double* x, double* y, double* result, int N
 * F2 : double* x, double* y, double* result, int N1, int N2
 * F3 : double* x, double* y, double* z, double* result, int N1, int N2, int N3
 */
using F1 = int(double *, double *, double *, int);
using F2 = int(double *, double *, double *, int, int);
using F3 = int(double *, double *, double *, int, int, int);

/*
 * 矩阵与常数之间四则运算的参数列表
 * F1 : double* x, double y, double* result, int N
 * F2 : double* x, double y, double* result, int N1, int N2
 * F3 : double* x, double y, double* result, int N1, int N2, int N3
 */
using F1_c = int(double *, double, double *, int);
using F2_c = int(double *, double, double *, int, int);
using F3_c = int(double *, double, double *, int, int, int);

/*
 * 数学函数
 * double* input, double* result, int N
 * double* input, double* result, int N1, int N2
 * double* input, double* result, int N1, int N2, int N3
 */
using M1 = int(double *, double *, int);
using M2 = int(double *, double *, int, int);
using M3 = int(double *, double *, int, int, int);

/*
 * 初始化函数
 * double* arr, int N
 * double* input, int N1, int N2
 * double* input, int N1, int N2, int N3
 */
using I1 = int(double *, double, int);
using I2 = int(double *, double, int, int);
using I3 = int(double *, double, int, int, int);


using F = std::variant<
        std::function<F1>,
        std::function<F2>,
        std::function<F3>,
        std::function<F1_c>,
        std::function<F2_c>,
        std::function<F3_c>,
        std::function<M1>,
        std::function<M2>,
        std::function<M3>,
        std::function<I1>,
        std::function<I2>,
        std::function<I3>>;




template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };

int main(void)
{
    double* a_host;
    double* b_host;
    double* result_host;

    double* a_dev;
    double* b_dev;
    double* result_dev;

    int N{10};
    a_host = (double *) malloc(N * sizeof(double));
    b_host = (double *) malloc(N * sizeof(double));
    result_host = (double *) malloc(N * sizeof(double));

    cudaMalloc(&a_dev, N * sizeof(double));
    cudaMalloc(&b_dev, N * sizeof(double));
    cudaMalloc(&result_dev, N * sizeof(double));

    for(int i=0; i < N; i++)
    {
        a_host[i] = 0.;
        b_host[i] = 10 - i;
        result_host[i] = 0.;
    }
    cudaMemcpy(a_dev, a_host, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(result_dev, result_host, N * sizeof(double), cudaMemcpyHostToDevice);
    // F f = add1;
    m1 cos1 = cos;
    F f = cos1;

    std::get<6>(f)(a_dev, result_dev, N);
    /*
    if(std::holds_alternative<std::function<M1>>(f))
    {
        std::get<std::function<M1>>(f)(a_dev, result_dev, N);
    }
    */




    cudaMemcpy(result_host, result_dev, N * sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i < N; i++)
    {
        std::cout << result_host[i] << '\t';
    }
    std::cout << '\n';
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(result_dev);
    free(a_host);
    free(b_host);
    free(result_host);    
}

