#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <io/netcdf>
#include <core/function/func.h>


int R(double* phi, double* result, int N1, int N2)
{
    laplace(phi, result, N1, N2);
    return EXIT_SUCCESS;
}


int phi1(double* phi, double* result, double h, int N1, int N2)
{
    R(phi, result, N1, N2);
    mul(result, h / 3, result, N1, N2);
    add(phi, result, result, N1, N2);
    return EXIT_SUCCESS;
}


int phi2(double* phi, double* result, double h, int N1, int N2)
{

    double* phi1_result;
    cudaMalloc(&phi1_result, N1 * N2 * sizeof(double));
    phi1(phi, phi1_result, h, N1, N2);
    R(phi1_result, result, N1, N2);
    mul(result, h / 2, result, N1, N2);
    add(phi, result, result, N1, N2);
    cudaFree(phi1_result);
    return EXIT_SUCCESS;
}


int next(double* phi, double* result, double h, int N1, int N2)
{
    double* phi2_result;
    cudaMalloc(&phi2_result, N1 * N2 * sizeof(double));  
    phi2(phi, phi2_result, h, N1, N2);
    R(phi2_result, result, N1, N2);
    mul(result, h, result, N1, N2);
    add(phi, result, result, N1, N2);
    cudaFree(phi2_result);
    return EXIT_SUCCESS;        
}


int main(void)
{
    int N1{64};
    int N2{48};
    double h{0.1};
    double* init_host;
    double* result_host;
    init_host = (double*) malloc(N1 * N2 * sizeof(double));   
    result_host = (double*) malloc(N1 * N2 * sizeof(double)); 
    double* init_dev;
    double* result_dev;
    cudaMalloc(&init_dev, N1 * N2 * sizeof(double));
    cudaMalloc(&result_dev, N1 * N2 * sizeof(double));


    char filepath[] = "./input.nc";
    char varname[] = "temperature";
    netcdf::ds(init_host, filepath, varname);

    cudaMemcpy(init_dev, init_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    cudaMemcpy(result_dev, result_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);

    for(int i=0; i<360; i++)
    {
        std::cout << "Round " << i + 1 << std::endl;
        next(init_dev, result_dev, h, N1, N2);
        //laplace_host(init_dev, result_dev, N1, N2);
        cudaMemcpy(init_dev, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToDevice);
        
        if((i + 1) % 10 == 0)
        {
            cudaMemcpy(result_host, result_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);
            std::ofstream outfile;
            outfile.open("./result/" + std::to_string(i + 1) + ".txt");
            for(int j=0; j < N1; j++)
            {
                for(int k=0; k <N2; k++)
                {
                    outfile << result_host[j * N2 + k] << '\t';
                }
                outfile << '\n';
            }
            outfile.close();
        }
    }
}