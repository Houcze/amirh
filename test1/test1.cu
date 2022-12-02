#include <map>
#include <string>
#include <tuple>
#include <iostream>
#include <chrono>
#include <core/mem.h>
#include <core/function/func.h>
#include <core/variables/Variables.h>


int main(void)
{
    int N1{6};
    int N2{12};
    Prop::shape s1{{"d1", N1}, {"d2", N2}};
    Npool.register_shape(s1);
    // Npool.register_seqlen(10);

    double *u_host;
    double *v_host;
    double *u_dev;
    double *v_dev;
    u_host = (double *)malloc(N1 * N2 * sizeof(double));
    v_host = (double *)malloc(N1 * N2 * sizeof(double));
    cudaMalloc(&u_dev, N1 * N2 * sizeof(double));
    cudaMalloc(&v_dev, N1 * N2 * sizeof(double));
    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N2; j++)
        {
            u_host[i * N2 + j] = 1.;
            v_host[i * N2 + j] = 2.;
        }
    }

    std::cout << "The size of s1 is " << Prop::size(s1) << std::endl;
    cudaMemcpy(u_dev, u_host, Prop::size(s1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v_dev, v_host, Prop::size(s1) * sizeof(double), cudaMemcpyHostToDevice);
    Variable U = make_tuple("u", s1, u_dev);
    Variable V = make_tuple("v", s1, v_dev);
    auto start1 = std::chrono::steady_clock::now();
    Variable UV = U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U;
    auto end1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = std::chrono::duration<double>(end1 - start1);
    std::cout << "The first call takes " << elapsed_seconds1.count() << std::endl;

    auto start2 = std::chrono::steady_clock::now();
    Variable U_V = U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U - U;
    auto end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = std::chrono::duration<double>(end2 - start2);
    std::cout << "The second call takes " << elapsed_seconds2.count() << std::endl;

    auto start3 = std::chrono::steady_clock::now();
    Variable uv = U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U + U;
    auto end3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds3 = std::chrono::duration<double>(end3 - start3);
    std::cout << "The third call takes " << elapsed_seconds3.count() << std::endl;

    double *result;
    result = (double *)malloc(N1 * N2 * sizeof(double));
    cudaMemcpy(result, std::get<double *>(uv), Prop::size(s1) * sizeof(double), cudaMemcpyDeviceToHost);

    Npool.print_variable_list();
    
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            std::cout << result[i * N2 + j] << '\t';
        }
        std::cout << '\n';
    }
    
    
}