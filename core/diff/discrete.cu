#define cuda_check_compute_result() {
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess){
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
    }
}


tensor zeta(tensor *u, tensor *v, tensor *output, double dx, double dy)
{
	// \zeta = (\partial{v}) / (\partial{x}) - (\partial{u}) / (\partial{y})
	// in wind format
    // 如何支持图模式
    tensor v_x = (3 * (*v) - 4 * (*v).move(-1, 0) + (*v).move(-2, 0)) / (2 * dy);
	tensor u_y = (3 * (*u) - 4 * (*u).move(0, -1) + (*u).move(0, -2)) / (2 * dx);
    tensor output = v_x - u_y;
}