#include <vector>
#include <cstdlib>
#include <iostream>
#include <malloc.h>
#include <io.cpp>
#include <typeinfo>
#include <cuda_runtime.h>
#include <cstdio>
#define MATHERROR 1
#define cuda_check_compute_result() { cudaError_t error = cudaGetLastError(); if(error!=cudaSuccess){ fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );}}

const static int tile_dim = 32;

typedef double (*FP)(double, double);

class array{
	public:
		/*
 		* 构造函数
 		* */
		array(char *, char *);
		array(double *, size_t *, size_t, size_t, double *);
		array(size_t, size_t, size_t *);
		//array(array);
		//~array();
		/*
 		* 四则运算
 		* */
		array operator+(array);
		array operator-(array);
		array operator*(array);
		array operator/(array);
		/*
 		* 常用的取形操作
 		* */
		int get_size();
		size_t* get_shape();
		size_t get_dims();
	
		int dataSync(double*);
		int gdataSync(double *);
		int get_shape(size_t*);

		/*
 		* 运算前的检查数组兼容性的函数
 		* */
		bool check(array *);
	private:
		size_t size{1};
		size_t dims;
		double *data;
		double *gdata;
		size_t *shape;
		
};


/*
* 两种类型的构造函数
*/
array::array(char *filepath, char *varname)
{
	// size = malloc_usable_size(data) / sizeof(double);	
	nc_get_size(&size, &dims, filepath, varname);
	data = (double *) std::malloc(size * sizeof(double));
	cudaMalloc(&gdata, size * sizeof(double));
	shape = (size_t *) std::malloc(dims * sizeof(size_t));
	nc_read_data(data, shape, filepath, varname);
	
	cudaMemcpy(gdata, data, size * sizeof(double), cudaMemcpyHostToDevice);
}


array::array(
	double *input_data, 
	size_t *input_shape, 
	size_t input_size, 
	size_t input_dims, 
	double *input_gdata
)
{
	size = input_size;
	dims = input_dims;
	data = (double *) std::malloc(size * sizeof(double));
	shape = (size_t *) std::malloc(dims * sizeof(size_t));
	std::memcpy(data, input_data, size * sizeof(double));
	std::memcpy(shape, input_shape, dims * sizeof(size_t));
	cudaMalloc(&gdata, size * sizeof(double));
	cudaMemcpy(gdata, input_gdata, size * sizeof(double), cudaMemcpyDeviceToDevice);
	
}

array::array(
	size_t input_size, 
	size_t input_dims, 
	size_t *input_shape
)
{
	size = input_size;
	dims = input_dims;
	std::memcpy(shape, input_shape, size * sizeof(size_t));
	data = (double *) std::malloc(size * sizeof(double));
	cudaMalloc(&gdata, size * sizeof(double));
}

/*
array::array(
	array base
)
{
	size = base.get_size();
	dims = base.get_dims();
	data = (double *) std::malloc(base.get_size() * sizeof(double));
	shape = (size_t *) std::malloc(base.get_dims() * sizeof(size_t));
	std::memcpy(data, base.dataSync(), base.get_size() * sizeof(double));
	std::memcpy(shape, base.get_shape(), base.get_dims() * sizeof(size_t));
	cudaMalloc(&gdata, base.get_size() * sizeof(double));
	cudaMemcpy(gdata, data, base.get_size() * sizeof(double), cudaMemcpyHostToDevice);
}
*/

// To destroy an array
/*
~array::array()
{
	cudaFree(gdata);
	free(data);
	free(shape);
}
*/

int array::get_size(){
	return size;
}

size_t* array::get_shape()
{	
	return shape;
}

size_t array::get_dims()
{
	return dims;
}

bool array::check(array *element)
{
	if((*element).get_size() == size)
	{
		return true;
	}
	return false;
}


int array::dataSync(double *var)
{
	std::memcpy(var, data, size * sizeof(double));
	return EXIT_SUCCESS;
}

int array::gdataSync(double *var)
{
	cudaMemcpy(var, gdata, size * sizeof(double), cudaMemcpyDeviceToDevice);
	return EXIT_SUCCESS;
}

int array::get_shape(size_t* var_shape)
{	
	std::memcpy(var_shape, shape, dims * sizeof(double));
	return EXIT_SUCCESS;
}

__device__ double __add__(double x, double y)
{
	return x + y;
}

__device__ double __sub__(double x, double y)
{
	return x - y;
}

__device__ double __mul__(double x, double y)
{
	return x * y;
}

__device__ double __div__(double x, double y)
{
	return x / y;
}

__device__ FP fp_add = __add__;
__device__ FP fp_sub = __sub__;
__device__ FP fp_mul = __mul__;
__device__ FP fp_div = __div__;


// 三维代数计算
__global__ void f3d(double *input1, double *input2, double *result, int d1, int d2, int d3, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int z_index = blockIdx.z * blockDim.z + threadIdx.z;	

	int index = x_index + y_index * d1 + z_index * d1 * d2;
	if(index < d1 * d2 * d3)
		result[index] = (*func)(input1[index], input2[index]);
}

__global__ void f3dc(double *input1, double *input2, double *result, int d1, int d2, int d3, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int z_index = blockIdx.z * blockDim.z + threadIdx.z;	

	int index = x_index + y_index * d1 + z_index * d1 * d2;
	if(index < d1 * d2 * d3)
		result[index] = (*func)(input1[index], (*input2));
}


// 二维代数计算
__global__ void f2d(double *input1, double *input2, double *result, int d1, int d2, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * d1;
	if(index < d1 * d2)
		result[index] = (*func)(input1[index], input2[index]);
}

__global__ void f2dc(double *input1, double *input2, double *result, int d1, int d2, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * d1;
	if(index < d1 * d2)
		result[index] = (*func)(input1[index], (*input2));
}

// 代数计算
__global__ void f1d(double *input1, double *input2, double *result, int d1, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = x_index;
	if(index < d1)
		result[index] = (*func)(input1[index], input2[index]);
}


__global__ void f1dc(double *input1, double *input2, double *result, int d1, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = x_index;
	if(index < d1)
		result[index] = (*func)(input1[index], (*input2));
}

////////

int 
alg(
	double *input1, 
	double *input2, 
	double *result, 
	size_t dims, 
	size_t size,
	size_t* shape, 
	double (*func)(double, double)
)
{
	size_t d1 = shape[0];
	size_t d2;
	size_t d3;
	switch(dims){
		case 1:
			f1d<<<dim3(d1 / tile_dim + 1), dim3(tile_dim)>>>(input1, input2, result, d1, func);
			cudaDeviceSynchronize();
			break;
		case 2:
			{
				d2 = shape[1];
				f2d<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1), dim3(tile_dim, tile_dim)>>>(input1, input2, result, d1, d2, func);
				cudaDeviceSynchronize();
				cudaError_t error = cudaGetLastError();
				if(error!=cudaSuccess)
				{
					fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
					exit(-1);
				}			
				break;
			}
		case 3:
			d2 = shape[1];
			d3 = shape[2];
			f3d<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1, d3 / tile_dim + 1), dim3(tile_dim, tile_dim, tile_dim)>>>(input1, input2, result, d1, d2, d3, func);
			cudaDeviceSynchronize();
			break;
	}	
	return EXIT_SUCCESS;
}


int 
broadcast(
	double *input1, 
	double *input2, 
	double *result, 
	size_t dims, 
	size_t size,
	size_t* shape, 
	double (*func)(double, double)
)
{
	size_t d1 = shape[0];
	size_t d2;
	size_t d3;

	double *dcoff;
	cudaMalloc(&dcoff, 1 * sizeof(double));
	cudaMemcpy(dcoff, input2, 1 * sizeof(double), cudaMemcpyHostToDevice);
	
	switch(dims){
		case 1:
			f1dc<<<dim3(d1 / tile_dim + 1), dim3(tile_dim)>>>(input1, dcoff, result, d1, func);
			cudaDeviceSynchronize();
			cudaFree(dcoff);
			break;
		case 2:
			{
				d2 = shape[1];
				f2dc<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1), dim3(tile_dim, tile_dim)>>>(input1, dcoff, result, d1, d2, func);
				cudaDeviceSynchronize();
				cudaFree(dcoff);
				cudaError_t error = cudaGetLastError();
				if(error!=cudaSuccess)
				{
					fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
					exit(-1);
				}
				break;
			}		
		case 3:
			d2 = shape[1];
			d3 = shape[2];
			f3dc<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1, d3 / tile_dim + 1), dim3(tile_dim, tile_dim, tile_dim)>>>(input1, dcoff, result, d1, d2, d3, func);
			cudaDeviceSynchronize();
			cudaFree(dcoff);
			break;
	}	
	return EXIT_SUCCESS;
}

array array::operator+(array element)
{
    if(check(&element))
    {
        // Check funciton runs a size and shape check
        double *result;
        double *gresult;
		double *bdata;
		cudaMalloc(&bdata, size * sizeof(double));

		element.gdataSync(bdata);
		result = (double *) std::malloc(size * sizeof(double));
        cudaMalloc(&gresult, size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_add, sizeof(FP));
        alg(gdata, bdata, gresult, dims, size, shape, fp_h);
        cudaMemcpy(result, gresult, size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(gdata);
        return array(result, shape, size, dims, gresult);
    }
	
}

array array::operator-(array element)
{
    if(check(&element))
    {
        // Check funciton runs a size and shape check
        double *result;
        double *gresult;
		double *bdata;
		cudaMalloc(&bdata, size * sizeof(double));

		element.gdataSync(bdata);
		result = (double *) std::malloc(size * sizeof(double));
        cudaMalloc(&gresult, size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_sub, sizeof(FP));
        alg(gdata, bdata, gresult, dims, size, shape, fp_h);
        cudaMemcpy(result, gresult, size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(gdata);
        return array(result, shape, size, dims, gresult);
    }
	
}


array array::operator*(array element)
{
    if(check(&element))
    {
        // Check funciton runs a size and shape check
        double *result;
        double *gresult;
		double *bdata;
		cudaMalloc(&bdata, size * sizeof(double));

		element.gdataSync(bdata);
		result = (double *) std::malloc(size * sizeof(double));
        cudaMalloc(&gresult, size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_mul, sizeof(FP));
        alg(gdata, bdata, gresult, dims, size, shape, fp_h);
        cudaMemcpy(result, gresult, size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(bdata);
        return array(result, shape, size, dims, gresult);
    }
	
}




array array::operator/(array element)
{
    if(check(&element))
    {
        // Check funciton runs a size and shape check
        double *result;
        double *gresult;
		double *bdata;
		cudaMalloc(&bdata, size * sizeof(double));

		element.gdataSync(bdata);
		result = (double *) std::malloc(size * sizeof(double));
        cudaMalloc(&gresult, size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_div, sizeof(FP));
        alg(gdata, bdata, gresult, dims, size, shape, fp_h);
        cudaMemcpy(result, gresult, size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(gdata);
        return array(result, shape, size, dims, gresult);
    }
	
}


array operator+(array _this, double coff)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_add, sizeof(FP));
	broadcast(gdata, &coff, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}

array operator+(double coff, array _this)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_add, sizeof(FP));
	broadcast(gdata, &coff, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}

array operator-(array _this, double coff)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_sub, sizeof(FP));
	broadcast(gdata, &coff, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}

array operator-(double coff, array _this)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_sub, sizeof(FP));
	broadcast(gdata, &coff, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}

// mul with a coff
array operator*(array _this, double coff)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_mul, sizeof(FP));
	broadcast(gdata, &coff, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}

array operator*(double coff, array _this)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_mul, sizeof(FP));
	broadcast(gdata, &coff, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}

array operator/(array _this, double coff)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_div, sizeof(FP));
	broadcast(gdata, &coff, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}

array operator/(double coff, array _this)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_div, sizeof(FP));
	broadcast(gdata, &coff, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}


// 完成中央差分的基础函数
__global__ void bias_i(double *input, double *output, int width, int height, int i)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * width + x_index;

	if(((index / height) >= i) && ((index / height) < (width + i)))
		output[index - i * height] = input[index];
	
}


__global__ void bias_j(double *input, double *output, int width, int height, int j)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y_index * width + x_index;

    if(((index % height) >= j) && ((index % height) < (height + j)))
        output[index - j] = input[index];
	
}


array bias(array base, int i, int j)
{
	/*
	* 该函数只针对2d数组进行定义，但是不检查数组形状
	*/
	double *result;
	double *gresult_i;
	double *gresult_j;

	result = (double *) std::malloc(base.get_size() * sizeof(double));
	cudaMalloc(&gresult_i, base.get_size() * sizeof(double));
	cudaMalloc(&gresult_j, base.get_size() * sizeof(double));

	auto shape = base.get_shape();
	size_t d1 = shape[0];
	size_t d2 = shape[1];
	/*
 	* 注意此处的调用未完成 
 	* */
 	double *gdata;
	cudaMalloc(&gdata, base.get_size() * sizeof(double));
	base.gdataSync(gdata);
	
	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_add, sizeof(FP));

	bias_i<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1), dim3(tile_dim, tile_dim)>>>(gdata, gresult_i, d1, d2, i);
	bias_j<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1), dim3(tile_dim, tile_dim)>>>(gresult_i, gresult_j, d1, d2, j);
	//f2d<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1), dim3(tile_dim, tile_dim)>>>(gresult_i, gresult_j, gresult, d1, d2, fp_h);
	cudaMemcpy(result, gresult_j, base.get_size() * sizeof(double), cudaMemcpyDeviceToHost);
	array output=array(result, base.get_shape(), base.get_size(), base.get_dims(), gresult_j);
	
	cudaFree(gresult_i);
	cudaFree(gresult_j);
	free(result);
	return output;
}

/*
double array::max()
{
	double m_val{0.};
	for(int i=0; i<d1; i++)
	{
		for(int j=0; j<d2; j++)
		{
			for(int k=0; k<d3; k++)
			{
				if(m_val < data[i][j][k])
				{
					m_val = data[i][j][k];
				}
			}
		}
	}
	return m_val;
}
*/

__global__ void mf3d(double *input, double *output, int d1, int d2, int d3, double (*func)(double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int z_index = blockIdx.z * blockDim.z + threadIdx.z;	

	int index = x_index + y_index * d1 + z_index * d1 * d2;
	if(index < d1 * d2 * d3)
		output[index] = (*func)(input[index]);
}


__global__ void mf2d(double *input, double *output, int d1, int d2, double (*func)(double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * d1;
	if(index < d1 * d2)
		output[index] = (*func)(input[index]);
}


__global__ void mf1d(double *input, double *output, int d1, double (*func)(double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = x_index;
	if(index < d1)
		output[index] = (*func)(input[index]);
}


int 
broadcast(
	double *input,  
	double *output, 
	size_t dims, 
	size_t size,
	size_t* shape, 
	double (*func)(double)
)
{
	size_t d1 = shape[0];
	size_t d2;
	size_t d3;
	
	switch(dims){
		case 1:
			mf1d<<<dim3(d1 / tile_dim + 1), dim3(tile_dim)>>>(input, output, d1, func);
			cudaDeviceSynchronize();
	
			break;
		case 2:
			{
				d2 = shape[1];
				mf2d<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1), dim3(tile_dim, tile_dim)>>>(input, output, d1, d2, func);
				cudaDeviceSynchronize();
			
				cudaError_t error = cudaGetLastError();
				if(error!=cudaSuccess)
				{
					fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
					exit(-1);
				}
				break;
			}		
		case 3:
			d2 = shape[1];
			d3 = shape[2];
			mf3d<<<dim3(d1 / tile_dim + 1, d2 / tile_dim + 1, d3 / tile_dim + 1), dim3(tile_dim, tile_dim, tile_dim)>>>(input, output, d1, d2, d3, func);
			cudaDeviceSynchronize();
			break;
	}	
	return EXIT_SUCCESS;
}

typedef double (*FP1var)(double);
__device__ double dsin(double x) {return sin(x);}
__device__ double dcos(double x) {return cos(x);}
__device__ FP1var fp_sin = dsin;
__device__ FP1var fp_cos = dcos;

array sin(array _this)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double *data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double *gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP1var fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_sin, sizeof(FP));
	broadcast(gdata, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}

array cos(array _this)
{
	double *result;
	double *gresult;
	
	result = (double *) std::malloc(_this.get_size() * sizeof(double));
	cudaMalloc(&gresult, _this.get_size() * sizeof(double));
	
	double* data;
	data = (double *) std::malloc(_this.get_size() * sizeof(double)); 

	_this.dataSync(data);

	size_t* shape;
	shape = (size_t *) std::malloc(_this.get_dims() * sizeof(size_t));
	
	_this.get_shape(shape);
	double* gdata;
	
	cudaMalloc(&gdata, _this.get_size() * sizeof(double));
	int status = _this.gdataSync(gdata);
	//cudaMemcpy(gdata, data, _this.get_size() * sizeof(double), cudaMemcpyHostToDevice);

	FP1var fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_cos, sizeof(FP));
	broadcast(gdata, gresult, _this.get_dims(), _this.get_size(), shape, fp_h);
	cudaMemcpy(result, gresult, _this.get_size() * sizeof(double), cudaMemcpyDeviceToHost);

	array output = array(result, _this.get_shape(), _this.get_size(), _this.get_dims(), gresult);
	cudaFree(gresult);
	free(result);
	return output;
}







