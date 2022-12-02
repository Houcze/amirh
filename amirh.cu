#include <string>
#include <fstream>
#include <iostream>
#include <io/netcdf>
#include <core/mem.h>
#include <cuda_runtime.h>
#include <core/function/func.h>
#include <core/variables/Variables.h>

const double d = 1;  // d是分辨率网格距离
const double m = 111194.92664455873;
const double omega = 7.27220521664304e-05;
const double g = 9.80665;

/*
 *  偏移规律：左负右正 上负下正
 *          9
 *      6   2   5
 *  10  3   0   1   12
 *      7   4   8
 *          11
 * Computation domain & grid system
 * 0 -> (0, 0)
 * 1 -> (0, 1)
 * 2 -> (-1, 0)
 * 3 -> (0, -1)
 * 4 -> (1, 0)
 * 5 -> (-1, 1)
 * 6 -> (-1, -1)
 * 7 -> (1, -1)
 * 8 -> (1, 1)
 * 9 -> (-2, 0)
 * 10 -> (0, -2)
 * 11 -> (2, 0)
 * 12 -> (0, 2)
*/


Variable Jpp(Variable A, Variable B)
{
/*  Jpp(A, B) = ((A1 - A3)(B2 - B4)-(A2 - A4)(B1 - B3)) / 4d^2
 *  偏移规律：左负右正 上负下正
 *          9
 *      6   2   5
 *  10  3   0   1   12
 *      7   4   8
 *          11
 * Computation domain & grid system
 * 0 -> (0, 0)
 * 1 -> (0, 1)
 * 2 -> (-1, 0)
 * 3 -> (0, -1)
 * 4 -> (1, 0)
 * 5 -> (-1, 1)
 * 6 -> (-1, -1)
 * 7 -> (1, -1)
 * 8 -> (1, 1)
 * 9 -> (-2, 0)
 * 10 -> (0, -2)
 * 11 -> (2, 0)
 * 12 -> (0, 2)
*/
    /*
    * result1 = A1 - A3
    * result2 = B2 - B4
    * result3 = A2 - A4
    * result4 = B1 - B3
    */
    Variable result = (parti_diff(A, 0, 1) - parti_diff(A, 0, -1)) * 
                      (parti_diff(B, -1, 0) - parti_diff(B, 1, 0)) - 
                      (parti_diff(A, -1, 0) - parti_diff(A, 1, 0)) * 
                      (parti_diff(B, 0, 1) - parti_diff(B, 0, -1));

    result = result / (4 * d * d);
    return result;
}


Variable Jpm1(Variable A, Variable B)
{
/*  Jpm(A, B) = (A5(B2-B1)-A7(B3-B4)+A6(B3-B2)-A8(B4-B1)) / 4d^2
 *  偏移规律：左负右正 上负下正
 *          9
 *      6   2   5
 *  10  3   0   1   12
 *      7   4   8
 *          11
 * Computation domain & grid system
 * 0 -> (0, 0)
 * 1 -> (0, 1)
 * 2 -> (-1, 0)
 * 3 -> (0, -1)
 * 4 -> (1, 0)
 * 5 -> (-1, 1)
 * 6 -> (-1, -1)
 * 7 -> (1, -1)
 * 8 -> (1, 1)
 * 9 -> (-2, 0)
 * 10 -> (0, -2)
 * 11 -> (2, 0)
 * 12 -> (0, 2)
*/
    // Jpm(A, B) = (A5(B2-B1)-A7(B3-B4)+A6(B3-B2)-A8(B4-B1)) / 4d^2
    Variable result = parti_diff(A, -1, 1) * (parti_diff(B, -1, 0) - parti_diff(B, 0, 1)) - 
                      parti_diff(A, 1, -1) * (parti_diff(B, 0, -1) - parti_diff(B, 1, 0)) + 
                      parti_diff(A, -1, -1) * (parti_diff(B, 0, -1) - parti_diff(B, -1, 0)) - 
                      parti_diff(A, 1, 1) * (parti_diff(B, 1, 0) - parti_diff(B, 0, 1));

    result = result / (4 * d * d);

    return result;
}

Variable Jmp1(Variable A, Variable B)
{
/*  Jmp(A, B) = (A1(B5-B8)-A3(B6-B7)-A2(B5-B6)+A4(B8-B7)) / 4d^2
 *  偏移规律：左负右正 上负下正
 *          9
 *      6   2   5
 *  10  3   0   1   12
 *      7   4   8
 *          11
 * Computation domain & grid system
 * 0 -> (0, 0)
 * 1 -> (0, 1)
 * 2 -> (-1, 0)
 * 3 -> (0, -1)
 * 4 -> (1, 0)
 * 5 -> (-1, 1)
 * 6 -> (-1, -1)
 * 7 -> (1, -1)
 * 8 -> (1, 1)
 * 9 -> (-2, 0)
 * 10 -> (0, -2)
 * 11 -> (2, 0)
 * 12 -> (0, 2)
*/
    Variable result = parti_diff(A, 0, 1) * (parti_diff(B, -1, 1) - parti_diff(B, 1, 1)) - 
                      parti_diff(A, 0, -1) * (parti_diff(B, -1, -1) - parti_diff(B, 1, -1)) - 
                      parti_diff(A, -1, 0) * (parti_diff(B, -1, 1) - parti_diff(B, -1, -1)) + 
                      parti_diff(A, 1, 0) * (parti_diff(B, 1, 1) - parti_diff(B, 1, -1));

    result = result / (4 * d * d);
    return result;
}

Variable Jmm(Variable A, Variable B)
{
/*  Jmm(A, B) = ((A5-A7)(B6-B8)-(A6-A8)(B5-B7)) / 8d^2
 *  偏移规律：左负右正 上负下正
 *          9
 *      6   2   5
 *  10  3   0   1   12
 *      7   4   8
 *          11
 * Computation domain & grid system
 * 0 -> (0, 0)
 * 1 -> (0, 1)
 * 2 -> (-1, 0)
 * 3 -> (0, -1)
 * 4 -> (1, 0)
 * 5 -> (-1, 1)
 * 6 -> (-1, -1)
 * 7 -> (1, -1)
 * 8 -> (1, 1)
 * 9 -> (-2, 0)
 * 10 -> (0, -2)
 * 11 -> (2, 0)
 * 12 -> (0, 2)
*/
    Variable result = (parti_diff(A, -1, 1) - parti_diff(A, 1, -1)) * 
                      (parti_diff(B, -1, -1) - parti_diff(B, 1, 1)) - 
                      (parti_diff(A, -1, -1) - parti_diff(A, 1, 1)) * 
                      (parti_diff(B, -1, 1) - parti_diff(B, 1, -1));

    result = result / (8 * d * d);
    return result;   

}


Variable Jmp2(Variable A, Variable B)
{
/*  Jmp(A, B) = (A5(B9-B12)-A7(B10-B1)-A6(B9-B10)+A8(B12-B11)) / 8d^2
 *  偏移规律：左负右正 上负下正
 *          9
 *      6   2   5
 *  10  3   0   1   12
 *      7   4   8
 *          11
 * Computation domain & grid system
 * 0 -> (0, 0)
 * 1 -> (0, 1)
 * 2 -> (-1, 0)
 * 3 -> (0, -1)
 * 4 -> (1, 0)
 * 5 -> (-1, 1)
 * 6 -> (-1, -1)
 * 7 -> (1, -1)
 * 8 -> (1, 1)
 * 9 -> (-2, 0)
 * 10 -> (0, -2)
 * 11 -> (2, 0)
 * 12 -> (0, 2)
*/

    Variable result = parti_diff(A, -1, 1) * (parti_diff(B, -2, 0) - parti_diff(B, 0, 2)) - 
                      parti_diff(A, 1, -1) * (parti_diff(B, 0, -2) - parti_diff(B, -2, 0)) - 
                      parti_diff(A, -1, -1) * (parti_diff(B, -2, 0) - parti_diff(B, 0, -2)) + 
                      parti_diff(A, 1, 1) * (parti_diff(B, 0, 2) - parti_diff(B, 2, 0));

    result = result / (8 * d * d);

    return result;
}


Variable Jpm2(Variable A, Variable B)
{
/*  Jpm(A, B) = (A9(B6-B5)-A11(B7-B8)+A10(B7-B6)-A12(B8-B5)) / 8d^2
 *  偏移规律：左负右正 上负下正
 *          9
 *      6   2   5
 *  10  3   0   1   12
 *      7   4   8
 *          11
 * Computation domain & grid system
 * 0 -> (0, 0)
 * 1 -> (0, 1)
 * 2 -> (-1, 0)
 * 3 -> (0, -1)
 * 4 -> (1, 0)
 * 5 -> (-1, 1)
 * 6 -> (-1, -1)
 * 7 -> (1, -1)
 * 8 -> (1, 1)
 * 9 -> (-2, 0)
 * 10 -> (0, -2)
 * 11 -> (2, 0)
 * 12 -> (0, 2)
*/
    Variable result = parti_diff(A, -2, 0) * (parti_diff(B, -1, -1) - parti_diff(B, -1, 1)) - 
                      parti_diff(A, 2, 0) * (parti_diff(B, 1, -1) - parti_diff(B, 1, 1)) + 
                      parti_diff(A, 0, -2) * (parti_diff(B, 1, -1) - parti_diff(B, -1, -1)) - 
                      parti_diff(A, 0, 2) * (parti_diff(B, 1, 1) - parti_diff(B, -1, 1));

    result = result / (8 * d * d);
    return result;
}

Variable J1(Variable A, Variable B)
{
    /*
    * J1(A, B) = \frac{1}{3}(Jpp(A, B)+Jpm(A,B)+Jmp(A, B))
    */
    Variable result = (Jpp(A, B) + Jpm1(A, B) + Jmp1(A, B)) / 3;
    return result;
}

Variable J2(Variable A, Variable B)
{
    /*
     * J2(A, B) = \frac{1}{3}(Jmm(A, B) + Jmp(A, B) + Jpm(A, B))
    */
    Variable result = (Jmm(A, B) + Jmp2(A, B) + Jpm2(A, B)) / 3;

    return result;
}

Variable J(Variable A, Variable B)
{
	// J(A, B) = 2J1(A, B) - J2(A ,B)
    Variable result = 2 * J1(A, B) - J2(A, B);
    return result;
}

Variable R(Variable phi)
{
    Variable result = parti_diff(phi, 1, 0) + parti_diff(phi, -1, 0) + parti_diff(phi, 0, 1) + parti_diff(phi, 0, -1) - 4 * phi;
    return result;
}

Variable phi1(Variable phi, double h)
{
    Variable result = R(phi);
    result = result * (h / 3);
    result = result + phi;
    return result;
}

Variable phi2(Variable phi, double h)
{
    Variable result = phi1(phi, h);
    result = R(result);
    result = result * (h / 2);
    result = result + phi;
    return result;
}

/*
Variable next(Variable phi, double h)
{   
    Variable result = phi2(phi);
    result = R(phi);
    result = result * h;
    result = result + phi;
    return result;       
}
*/

Variable next(Variable pz_pt, Variable z, double h, Variable lat, Variable lon)
{
    Variable j_result = (parti_diff(pz_pt, 1, 0) + parti_diff(pz_pt, -1, 0) + parti_diff(pz_pt, 0, 1) + parti_diff(pz_pt, 0, -1));
    Variable coeff = g / (sin(lat * 3.1415926 / 180) * 2 * omega);
    Variable zz = laplace(z);
    //zero_boundary(zz);
    z = z + (pz_pt) * (h / 2);
    Variable result = j_result + J(zz * coeff + (sin(lat * 3.1415926 / 180) * 2 * omega), z  + pz_pt * (h / 2)) * (-d * d / (m * m));
    result = result * 0.25;
    return result;
}


int main(void)
{   
    int t{1};
    int N1{41};
    int N2{360};
    double h{60 * 60};
    Prop::shape s0 = {{"d1", N1}, {"d2", N2}};
    double* init_z_host_1;
    double* init_z_host_2;
    double* result_z_host;
    double* init_pzpt_host;
    //double* result_pzpt_host;
    
    init_z_host_1 = (double*) malloc(t * N1 * N2 * sizeof(double));
    init_z_host_2 = (double*) malloc(t * N1 * N2 * sizeof(double));   
    result_z_host = (double*) malloc(N1 * N2 * sizeof(double));
    init_pzpt_host = (double*) malloc(N1 * N2 * sizeof(double));   
    //result_pzpt_host = (double*) malloc(N1 * N2 * sizeof(double));  
    
    double* init_z_dev_1;
    double* init_z_dev_2;
    double* result_z_dev;
    double* init_pzpt_dev;
    double* result_pzpt_dev;

    cudaMalloc(&init_z_dev_1, N1 * N2 * sizeof(double));
    cudaMalloc(&init_z_dev_2, N1 * N2 * sizeof(double));

    cudaMalloc(&result_z_dev, N1 * N2 * sizeof(double));
    cudaMalloc(&init_pzpt_dev, N1 * N2 * sizeof(double));
    cudaMalloc(&result_pzpt_dev, N1 * N2 * sizeof(double));


    char filepath_1[] = "./ds/amirh_20200531_06_00.nc";
    char filepath_2[] = "./ds/amirh_20200531_12_00.nc";
    char varname[] = "HGT_500mb";
    netcdf::ds(init_z_host_1, filepath_1, varname);
    netcdf::ds(init_z_host_2, filepath_2, varname);

    double* lat_host;
    double* lat_d_dev;
    double* lon_host;
    double* lon_d_dev;

    lat_host = (double*) malloc(N1 * sizeof(double));
    lon_host = (double*) malloc(N2 * sizeof(double));
    cudaMalloc(&lat_d_dev, sizeof(double) * N1 * N2);
    cudaMalloc(&lon_d_dev, sizeof(double) * N2 * N2);
    char lon_s[] = "longitude";
    char lat_s[] = "latitude";
    netcdf::ds(lat_host, filepath_1, lat_s);
    netcdf::ds(lon_host, filepath_2, lon_s);
    double* lat_d_host;
    double* lon_d_host;
    lat_d_host = (double*) malloc(N1 * N2 * sizeof(double));
    lon_d_host = (double*) malloc(N1 * N2 * sizeof(double));
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            lat_d_host[i * N2 + j] = lat_host[i];
            lon_d_host[i * N2 + j] = lon_host[j];
        }
    }
    cudaMemcpy(lat_d_dev, lat_d_host, N1 * N2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(lon_d_dev, lon_d_host, N1 * N2 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(init_z_dev_1, init_z_host_1, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);
    cudaMemcpy(init_z_dev_2, init_z_host_2, sizeof(double) * N1 * N2, cudaMemcpyHostToDevice);

    sub(init_z_dev_2, init_z_dev_1, init_pzpt_dev, N1, N2);
    div(init_pzpt_dev, 60 * 60 * 6, init_pzpt_dev, N1, N2);
    /* 注意上面这个h还需要修改 */

    cudaMemcpy(init_pzpt_host, init_pzpt_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);

    Variable pzpt = std::make_tuple("pzpt", s0, init_pzpt_dev);
    Variable z = std::make_tuple("z", s0, init_z_dev_2);
    
    for(int i=0; i<24; i++)
    {
        std::cout << "Round " << i + 1 << std::endl;        
        pzpt = next(pzpt, z, h, std::make_tuple("lat", s0, lat_d_dev), std::make_tuple("lon", s0, lon_d_dev));
        /*
        if(i >= 0)
        {
            cudaMemcpy(result_z_host, init_z_dev_2, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);
            to_txt(result_z_host, "./result/" + std::to_string(i + 1), N1, N2);
        }
        */
    }
}
