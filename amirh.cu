#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <io/netcdf>
#include <cuda_runtime.h>
#include <core/Ops/Node.h>
#include <core/function/func.h>


int f()
{
    Mul u2(u, u);
    Tan tanPhi(phi);
    Mul u2tanPhi(u2, tanphi);
    Div frac_u2tanPhi_r(u2tanPhi, r);

    Mul vw(v, w);
    Div frac_vw_r(v_w, r);
    Add frac_u2tanPhi_r_a_frac_vw_r(frac_u2tanPhi_r, frac_vw_r);

    Parti pPpPhi(p, phi);
    Div frac_pPpPhi_r(pPpPhi, r);
    Div frac_pPpPhi_rrho(frac_pPpPhi_r, rho);
    
    Sub frac_pPpPhi_rrho_sub_frac_u2tanPhi_r_a_frac_vw_r(frac_pPpPhi_rrho, frac_u2tanPhi_r_a_frac_vw_r);

    Mul fu(f, u);
    Sub frac_pPpPhi_rrho_sub_frac_u2tanPhi_r_a_frac_vw_r_sub_fu(frac_pPpPhi_rrho_sub_frac_u2tanPhi_r_a_frac_vw_r, fu);

    Output output(frac_pPpPhi_rrho_sub_frac_u2tanPhi_r_a_frac_vw_r_sub_fu);

    /*
     * 注册常数
    */
    Ops<I1> u(set_cons);
    Ops<I1> r(set_cons);
    Ops<I1> phi(set_cons);
    
    
    Ops<F2> u2(mul);
    Ops<M1> tanPhi(tan);
    Ops<F2> u2tanPhi(mul);
    Ops<F2> frac_u2tanPhi_r(div);

    /*
     * 注册计算树
    */
    u2.register_input_node(&u);
    u2.register_input_node(&u);
    tanPhi.register_input_node(&phi);    
    u2tanPhi.register_input_node(&u2);
    u2tanPhi.register_input_node(&tanPhi);
    frac_u2tanPhi_r.register_input_node(&u2tanPhi);
    frac_u2tanPhi_r.register_input_node(&r);


    Ops<I1> v(set_cons);
    Ops<I2> w(set_cons);
    
    Ops<F2> vw(mul);
    Ops<F2> frac_vw_r(div);
    Ops<F2> frac_u2tanPhi_r_a_frac_vw_r(add);

    vw.register_input_node(&v);
    vw.register_input_node(&w);

    frac_vw_r.register_input_node(&vw);
    frac_vw_r.register_input_node(&r);

    frac_u2tanPhi_r_a_frac_vw_r.register_input_node(&frac_u2tanPhi_r);
    frac_u2tanPhi_r_a_frac_vw_r.register_input_node(&frac_vw_r);

    /*
     * 球面坐标系偏微分如何离散化
    */

}

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
    int t{56};
    int N1{241};
    int N2{480};
    double h{0.1};
    double* init_host;
    double* result_host;
    init_host = (double*) malloc(t * N1 * N2 * sizeof(double));   
    result_host = (double*) malloc(N1 * N2 * sizeof(double)); 
    double* init_dev;
    double* result_dev;
    cudaMalloc(&init_dev, N1 * N2 * sizeof(double));
    cudaMalloc(&result_dev, N1 * N2 * sizeof(double));


    char filepath[] = "";
    char varname[] = "uvb";
    netcdf::ds(init_host, filepath, varname);

    cudaMemcpy(init_dev, init_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    cudaMemcpy(result_dev, result_host, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);

    for(int i=0; i<36000; i++)
    {
        std::cout << "Round " << i + 1 << std::endl;
        next(init_dev, result_dev, h, N1, N2);
        // laplace_host(init_dev, result_dev, N1, N2);
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