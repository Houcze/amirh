#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <io/netcdf>
#include <fstream>
#include <core/function/func.h>

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


int Jpp(double* A, double* B, double* output, int N1, int N2)
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
    double* result1;
    double* result2;
    double* result3;
    double* result4;

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));  
    cudaMalloc(&result3, N1 * N2 * sizeof(double));  
    cudaMalloc(&result4, N1 * N2 * sizeof(double));  

    cudaMemset(result1, 0, N1 * N2 * sizeof(double));
    cudaMemset(result2, 0, N1 * N2 * sizeof(double));
    cudaMemset(result3, 0, N1 * N2 * sizeof(double));
    cudaMemset(result4, 0, N1 * N2 * sizeof(double));

    cudaMemset(output, 0, N1 * N2 * sizeof(double));

    /*
    * result1 = A1 - A3
    * result2 = B2 - B4
    * result3 = A2 - A4
    * result4 = B1 - B3
    */
    parti_diff_add(A, result1, N1, N2, 0, 1);
    parti_diff_sub(A, result1, N1, N2, 0, -1);

    parti_diff_add(B, result2, N1, N2, -1, 0);
    parti_diff_sub(B, result2, N1, N2, 1, 0);
    
    parti_diff_add(A, result3, N1, N2, -1, 0);
    parti_diff_sub(A, result3, N1, N2, 1, 0);
    
    parti_diff_add(B, result4, N1, N2, 0, 1);
    parti_diff_sub(B, result4, N1, N2, 0, -1);

    mul(result1, result2, result1, N1, N2);
    mul(result3, result4, result3, N1, N2);

    sub(result1, result3, output, N1, N2);
    div(output, 4 * d * d, output, N1, N2);

    cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);
    cudaFree(result4);
    return EXIT_SUCCESS;
}


int Jpm1(double *A, double *B, double *output, int N1, int N2)
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
    double* result1;
    double* result2;
    double* result3;
    double* result4;

    double* A5;
    double* A7;
    double* A6;
    double* A8;
    cudaMalloc(&A5, N1 * N2 * sizeof(double));
    cudaMalloc(&A7, N1 * N2 * sizeof(double));
    cudaMalloc(&A6, N1 * N2 * sizeof(double));
    cudaMalloc(&A8, N1 * N2 * sizeof(double));

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));  
    cudaMalloc(&result3, N1 * N2 * sizeof(double));  
    cudaMalloc(&result4, N1 * N2 * sizeof(double));  

    cudaMemset(result1, 0, N1 * N2 * sizeof(double));
    cudaMemset(result2, 0, N1 * N2 * sizeof(double));
    cudaMemset(result3, 0, N1 * N2 * sizeof(double));
    cudaMemset(result4, 0, N1 * N2 * sizeof(double));

    cudaMemset(output, 0, N1 * N2 * sizeof(double));


    parti_diff_add(B, result1, N1, N2, -1, 0);
    parti_diff_sub(B, result1, N1, N2, 0, 1);


    parti_diff_add(B, result2, N1, N2, 0, -1);
    parti_diff_sub(B, result2, N1, N2, 1, 0);
    
    parti_diff_add(B, result3, N1, N2, 0, -1);
    parti_diff_sub(B, result3, N1, N2, -1, 0);
    
    parti_diff_add(B, result4, N1, N2, 1, 0);
    parti_diff_sub(B, result4, N1, N2, 0, 1);

    parti_diff(A, A5, N1, N2, -1, 1);
    parti_diff(A, A7, N1, N2, 1, -1);
    parti_diff(A, A6, N1, N2, -1, -1);
    parti_diff(A, A8, N1, N2, 1, 1);


    mul(result1, A5, result1, N1, N2);
    mul(result2, A7, result2, N1, N2);
    mul(result3, A6, result3, N1, N2);
    mul(result4, A8, result4, N1, N2);

    sub(result1, result2, result1, N1, N2);
    sub(result3, result4, result3, N1, N2);
    add(result1, result3, output, N1, N2);   
    div(output, 4 * d * d, output, N1, N2);

    cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);
    cudaFree(result4);
    cudaFree(A5);
    cudaFree(A7);
    cudaFree(A6);
    cudaFree(A8);

    return EXIT_SUCCESS;
}

int Jmp1(double *A, double *B, double *output, int N1, int N2)
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
    double* result1;
    double* result2;
    double* result3;
    double* result4;

    double* A1;
    double* A3;
    double* A2;
    double* A4;
    cudaMalloc(&A1, N1 * N2 * sizeof(double));
    cudaMalloc(&A3, N1 * N2 * sizeof(double));
    cudaMalloc(&A2, N1 * N2 * sizeof(double));
    cudaMalloc(&A4, N1 * N2 * sizeof(double));

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));  
    cudaMalloc(&result3, N1 * N2 * sizeof(double));  
    cudaMalloc(&result4, N1 * N2 * sizeof(double));  

    cudaMemset(result1, 0, N1 * N2 * sizeof(double));
    cudaMemset(result2, 0, N1 * N2 * sizeof(double));
    cudaMemset(result3, 0, N1 * N2 * sizeof(double));
    cudaMemset(result4, 0, N1 * N2 * sizeof(double));

    cudaMemset(output, 0, N1 * N2 * sizeof(double));


    parti_diff_add(B, result1, N1, N2, -1, 1);
    parti_diff_sub(B, result1, N1, N2, 1, 1);


    parti_diff_add(B, result2, N1, N2, -1, -1);
    parti_diff_sub(B, result2, N1, N2, 1, -1);
    
    parti_diff_add(B, result3, N1, N2, -1, 1);
    parti_diff_sub(B, result3, N1, N2, -1, -1);
    
    parti_diff_add(B, result4, N1, N2, 1, 1);
    parti_diff_sub(B, result4, N1, N2, 1, -1);

    parti_diff(A, A1, N1, N2, 0, 1);
    parti_diff(A, A3, N1, N2, 0, -1);
    parti_diff(A, A2, N1, N2, -1, 0);
    parti_diff(A, A4, N1, N2, -1, 0);


    mul(result1, A1, result1, N1, N2);
    mul(result2, A3, result2, N1, N2);
    mul(result3, A2, result3, N1, N2);
    mul(result4, A4, result4, N1, N2);

    sub(result1, result2, result1, N1, N2);
    sub(result3, result4, result3, N1, N2);
    sub(result1, result3, output, N1, N2);   
    div(output, 4 * d * d, output, N1, N2);

    cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);
    cudaFree(result4);
    cudaFree(A1);
    cudaFree(A2);
    cudaFree(A3);
    cudaFree(A4);

    return EXIT_SUCCESS;
}

int Jmm(double* A, double *B, double *output, int N1, int N2)
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
    double* result1;
    double* result2;
    double* result3;
    double* result4;

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));  
    cudaMalloc(&result3, N1 * N2 * sizeof(double));  
    cudaMalloc(&result4, N1 * N2 * sizeof(double));  

    cudaMemset(result1, 0, N1 * N2 * sizeof(double));
    cudaMemset(result2, 0, N1 * N2 * sizeof(double));
    cudaMemset(result3, 0, N1 * N2 * sizeof(double));
    cudaMemset(result4, 0, N1 * N2 * sizeof(double));

    cudaMemset(output, 0, N1 * N2 * sizeof(double));

    parti_diff_add(A, result1, N1, N2, -1, 1);
    parti_diff_sub(A, result1, N1, N2, 1, -1);

    parti_diff_add(B, result2, N1, N2, -1, -1);
    parti_diff_sub(B, result2, N1, N2, 1, 1);
    
    parti_diff_add(A, result3, N1, N2, -1, -1);
    parti_diff_sub(A, result3, N1, N2, 1, 1);
    
    parti_diff_add(B, result4, N1, N2, -1, 1);
    parti_diff_sub(B, result4, N1, N2, 1, -1);

    mul(result1, result2, result1, N1, N2);
    mul(result3, result4, result3, N1, N2);

    sub(result1, result3, output, N1, N2);
    div(output, 8 * d * d, output, N1, N2);

    cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);
    cudaFree(result4);
    return EXIT_SUCCESS;   

}


int Jmp2(double *A, double *B, double *output, int N1, int N2)
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
    double* result1;
    double* result2;
    double* result3;
    double* result4;

    double* A5;
    double* A7;
    double* A6;
    double* A8;
    cudaMalloc(&A5, N1 * N2 * sizeof(double));
    cudaMalloc(&A7, N1 * N2 * sizeof(double));
    cudaMalloc(&A6, N1 * N2 * sizeof(double));
    cudaMalloc(&A8, N1 * N2 * sizeof(double));

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));  
    cudaMalloc(&result3, N1 * N2 * sizeof(double));  
    cudaMalloc(&result4, N1 * N2 * sizeof(double));  

    cudaMemset(result1, 0, N1 * N2 * sizeof(double));
    cudaMemset(result2, 0, N1 * N2 * sizeof(double));
    cudaMemset(result3, 0, N1 * N2 * sizeof(double));
    cudaMemset(result4, 0, N1 * N2 * sizeof(double));

    cudaMemset(output, 0, N1 * N2 * sizeof(double));


    parti_diff_add(B, result1, N1, N2, -2, 0);
    parti_diff_sub(B, result1, N1, N2, 0, 2);


    parti_diff_add(B, result2, N1, N2, 0, -2);
    parti_diff_sub(B, result2, N1, N2, -2, 0);
    
    parti_diff_add(B, result3, N1, N2, -2, 0);
    parti_diff_sub(B, result3, N1, N2, 0, -2);
    
    parti_diff_add(B, result4, N1, N2, 0, 2);
    parti_diff_sub(B, result4, N1, N2, 2, 0);

    parti_diff(A, A5, N1, N2, -1, 1);
    parti_diff(A, A7, N1, N2, 1, -1);
    parti_diff(A, A6, N1, N2, -1, -1);
    parti_diff(A, A8, N1, N2, 1, 1);


    mul(result1, A5, result1, N1, N2);
    mul(result2, A7, result2, N1, N2);
    mul(result3, A6, result3, N1, N2);
    mul(result4, A8, result4, N1, N2);

    sub(result1, result2, result1, N1, N2);
    sub(result3, result4, result3, N1, N2);
    sub(result1, result3, output, N1, N2);   
    mul(output, 8 * d * d, output, N1, N2);

    cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);
    cudaFree(result4);
    cudaFree(A5);
    cudaFree(A7);
    cudaFree(A6);
    cudaFree(A8);

    return EXIT_SUCCESS;
}


int Jpm2(double *A, double *B, double *output, int N1, int N2)
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
  
    double* result1;
    double* result2;
    double* result3;
    double* result4;

    double* A9;
    double* A11;
    double* A10;
    double* A12;
    cudaMalloc(&A9, N1 * N2 * sizeof(double));
    cudaMalloc(&A11, N1 * N2 * sizeof(double));
    cudaMalloc(&A10, N1 * N2 * sizeof(double));
    cudaMalloc(&A12, N1 * N2 * sizeof(double));

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));  
    cudaMalloc(&result3, N1 * N2 * sizeof(double));  
    cudaMalloc(&result4, N1 * N2 * sizeof(double));  

    cudaMemset(result1, 0, N1 * N2 * sizeof(double));
    cudaMemset(result2, 0, N1 * N2 * sizeof(double));
    cudaMemset(result3, 0, N1 * N2 * sizeof(double));
    cudaMemset(result4, 0, N1 * N2 * sizeof(double));

    cudaMemset(output, 0, N1 * N2 * sizeof(double));


    parti_diff_add(B, result1, N1, N2, -1, -1);
    parti_diff_sub(B, result1, N1, N2, -1, 1);


    parti_diff_add(B, result2, N1, N2, 1, -1);
    parti_diff_sub(B, result2, N1, N2, 1, 1);
    
    parti_diff_add(B, result3, N1, N2, 1, -1);
    parti_diff_sub(B, result3, N1, N2, -1, -1);
    
    parti_diff_add(B, result4, N1, N2, 1, 1);
    parti_diff_sub(B, result4, N1, N2, -1, 1);

    parti_diff(A, A9, N1, N2, -2, 0);
    parti_diff(A, A11, N1, N2, 2, 0);
    parti_diff(A, A10, N1, N2, 0, -2);
    parti_diff(A, A12, N1, N2, 0, 2);


    mul(result1, A9, result1, N1, N2);
    mul(result2, A11, result2, N1, N2);
    mul(result3, A10, result3, N1, N2);
    mul(result4, A12, result4, N1, N2);

    sub(result1, result2, result1, N1, N2);
    sub(result3, result4, result3, N1, N2);
    add(result1, result3, output, N1, N2);   
    div(output, 8 * d * d, output, N1, N2);

    cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);
    cudaFree(result4);
    cudaFree(A9);
    cudaFree(A11);
    cudaFree(A10);
    cudaFree(A12);

    return EXIT_SUCCESS;

}

int J1(double *A, double *B, double *output, int N1, int N2)
{
    /*
    * J1(A, B) = \frac{1}{3}(Jpp(A, B)+Jpm(A,B)+Jmp(A, B))
    */
	double* result1;
    double* result2;
    double* result3;

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));
    cudaMalloc(&result3, N1 * N2 * sizeof(double));

	Jpp(A, B, result1, N1, N2);
	Jpm1(A, B, result2, N1, N2);
	Jmp1(A, B, result3, N1, N2);

    add(output, result1, output, N1, N2);
    add(output, result2, output, N1, N2);
    add(output, result3, output, N1, N2);

    div(output, 3., output, N1, N2);
	cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);

    return EXIT_SUCCESS;
}

int J2(double *A, double *B, double *output, int N1, int N2)
{
    /*
     * J2(A, B) = \frac{1}{3}(Jmm(A, B) + Jmp(A, B) + Jpm(A, B))
    */
	double* result1;
    double* result2;
    double* result3;

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));
    cudaMalloc(&result3, N1 * N2 * sizeof(double));

	Jmm(A, B, result1, N1, N2);
	Jmp2(A, B, result2, N1, N2);
	Jpm2(A, B, result3, N1, N2);

    add(output, result1, output, N1, N2);
    add(output, result2, output, N1, N2);
    add(output, result3, output, N1, N2);

    div(output, 3., output, N1, N2);
	cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);

    return EXIT_SUCCESS;
}


/*
int zeta(double *u, double *v, double *output, int N1, int N2)
{
	// \zeta = (\parti{v}) / (\parti{x}) - (\parti{u}) / (\parti{y})
	// in wind format

    double* v_x;
    double* u_y;
    cudaMalloc(&v_x, N1 * N2 * sizeof(double));
    cudaMalloc(&u_y, N1 * N2 * sizeof(double));

    double* result1;
    double* result2;
    double* result3;
    double* result4;

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));    
    cudaMalloc(&result3, N1 * N2 * sizeof(double));
    cudaMalloc(&result4, N1 * N2 * sizeof(double)); 

    parti_diff(v, result1, N1, N2, -1, 0);   
    parti_diff(v, result1, N1, N2, -2, 0);
    parti_diff(v, result1, N1, N2, 0, -1);   
    parti_diff(v, result1, N1, N2, 0, -2);

    expectation(v, result1, result2, v_x, 3, -4, 1, N1, N2);    
    expectation(v, result3, result4, u_y, 3, -4, 1, N1, N2);

    div(v_x, 2 * dy, v_x, N1, N2);
    div(u_y, 2 * dx, u_y, N1, N2);

    mul(v_x, u_y, output, N1, N2);

    cudaFree(v_x);
    cudaFree(u_y);
    cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);
    cudaFree(result4);
    return EXIT_SUCCESS;

}
*/



int J(double *A, double *B, double *output, int N1, int N2)
{
	// J(A, B) = 2J1(A, B) - J2(A ,B)
    double *j1;
    double *j2;
    cudaMalloc(&j1, N1 * N2 * sizeof(double));
    cudaMalloc(&j2, N1 * N2 * sizeof(double));

    cudaMemset(j1, 0, N1 * N2 * sizeof(double));
    cudaMemset(j2, 0, N1 * N2 * sizeof(double));

	J1(A, B, j1, N1, N2);
	J2(A, B, j2, N1, N2);
    
    expectation(j1, j2, output, 2, -1, N1, N2);
    cudaFree(j1);
    cudaFree(j2);
    return EXIT_SUCCESS;
}


int zero_boundary(double* result, int N1, int N2)
{
    double* result_host;
    result_host = (double*) malloc(N1 * N2 * sizeof(double));
    cudaMemcpy(result_host, result, N1 * N2 * sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i<N1; i++)
    {
        for(int j=0; j<N2; j++)
        {
            if((i == 0) || (j == 0) || (j == N2-1) || (i==N1-1))
            {
                result_host[i * N2 + j] = 0;
            }
        }
    }
    cudaMemcpy(result, result_host, N1 * N2 * sizeof(double), cudaMemcpyHostToDevice);
    free(result_host);
    return EXIT_SUCCESS;
}


int to_txt(double* var, std::string filename, int N1, int N2)
{
    std::ofstream outfile;
    outfile.open("./" + filename + ".txt");
    for(int j=0; j < N1; j++)
    {
        for(int k=0; k <N2; k++)
        {
            outfile << var[j * N2 + k] << '\t';
        }
        outfile << '\n';
    }
    outfile.close();
    return EXIT_SUCCESS;
}


int to_txt_dev(double* var, std::string filename, int N1, int N2)
{
    double* host;
    host = (double*) malloc(N1 * N2 * sizeof(double));
    cudaMemcpy(host, var, N1 * N2 * sizeof(double), cudaMemcpyDeviceToHost);
    std::ofstream outfile;
    outfile.open("./" + filename + ".txt");
    for(int j=0; j < N1; j++)
    {
        for(int k=0; k <N2; k++)
        {
            outfile << host[j * N2 + k] << '\t';
        }
        outfile << '\n';
    }
    outfile.close();
    free(host);
    return EXIT_SUCCESS;
}



/*--------------------------------------------*/

int next(double* pz_pt, double* z, double* result, double h, int N1, int N2, double* lat, double* lon)
{
    /*
    * This result is the next timeslot of pz_pt
    */
    double* result1;
    double* result2;
    double* result3;
    double* result4;
    double* j_result;

    cudaMalloc(&result1, N1 * N2 * sizeof(double));
    cudaMalloc(&result2, N1 * N2 * sizeof(double));
    cudaMalloc(&result3, N1 * N2 * sizeof(double));
    cudaMalloc(&result4, N1 * N2 * sizeof(double));
    cudaMalloc(&j_result, N1 * N2 * sizeof(double));

    parti_diff(pz_pt, result1, N1, N2, 1, 0);
    parti_diff(pz_pt, result2, N1, N2, -1, 0);
    parti_diff(pz_pt, result3, N1, N2, 0, 1);
    parti_diff(pz_pt, result4, N1, N2, 0, -1);

    add(result1, result2, result1, N1, N2);
    add(result3, result4, result3, N1, N2);
    add(result1, result3, result1, N1, N2);

    mul(lat, 3.1415926 / 180, lat, N1, N2);
    sin(lat, lat, N1, N2);
    mul(lat, 2 * omega, lat, N1, N2);

    double* coeff;
    cudaMalloc(&coeff, N1 * N2 * sizeof(double));
    set_cons(coeff, g, N1, N2);
    
    
    div(coeff, lat, coeff, N1, N2);
    /*
     * coeff里出现了无穷大
    */
    
    double* zz;
    cudaMalloc(&zz, N1 * N2 * sizeof(double));
    laplace(z, zz, N1, N2);
    mul(zz, coeff, zz, N1, N2);
    
    add(zz, lat, zz, N1, N2);
    expectation(z, pz_pt, z, 1, 2 * h, N1, N2);    
    
    J(zz, z, j_result, N1, N2);
    

    expectation(result1, j_result, result1, 1, -d * d / (m * m), N1, N2);
    mul(result1, 0.25, result, N1, N2);
    zero_boundary(result, N1, N2);
    cudaFree(result1);
    cudaFree(result2);
    cudaFree(result3);
    cudaFree(result4);
    cudaFree(j_result);
    cudaFree(coeff);
    return EXIT_SUCCESS;
}


int main(int argc, char* argv[])
{   
    int t{1};
    int N1{41};
    int N2{360};
    double h{60 * 60};

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

    for(int i=0; i<24; i++)
    {
        std::cout << "Round " << i + 1 << std::endl;
        next(init_pzpt_dev, init_z_dev_2, result_pzpt_dev, h, N1, N2, lat_d_dev, lon_d_dev);
        cudaMemcpy(init_pzpt_dev, result_pzpt_dev, sizeof(double) * N1 * N2, cudaMemcpyDeviceToDevice);
        
        if(i >= 0)
        {
            cudaMemcpy(result_z_host, init_z_dev_2, sizeof(double) * N1 * N2, cudaMemcpyDeviceToHost);
            to_txt(result_z_host, "./result/" + std::to_string(i + 1), N1, N2);
        }
    }
}