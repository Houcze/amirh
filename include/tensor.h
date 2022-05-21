#include <cstdlib>
#include <iostream>
#include <malloc.h>
#include <io.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>


#define cuda_check_compute_result() {cudaError_t error = cudaGetLastError(); if(error!=cudaSuccess){ fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );}}

const static int threadsPerBlock = 32;

typedef double (*FP)(double, double);


enum DEVICE {
	CPU=0, CUDA, HIP
};

namespace io
{
	class tensor
	{
		public:
			/*
			* 	tensor(char *path, char *var) creates array from file with path
			*	tensor(double * data, size_t *shape, size_t size, size_t size) creates array from designated C/C++ array on CPU or GPU
			*/
			//~tensor();
			/*
			* 	get_size will return
			*/
			//virtual int get_size();
			//virtual size_t* get_shape();
			//virtual size_t get_dims();
			//virtual int get_shape(size_t*);
			template <class T>
			bool check(T *);
			size_t size{1};
			size_t dims;
			size_t *shape;
			double *data;
			enum DEVICE dtype;
	};


	namespace cpu
	{
		/* 	io::cpu::tensor will inherit from io::tensor, and will implement 
		*	tensor to_gpu()
		*/
		class tensor : public io::tensor 
		{
			public:
				tensor():io::tensor(){
					this->dtype=CPU;
				}
				tensor(char*, char *);
				tensor(double *, size_t *, size_t, size_t);
				double *get_data();
				~tensor();

		};
	}

	namespace cuda
	{
		/* 	io::gpu::tensor will inherit from io::tensor, and will implement 
		*	tensor to_cpu()
		*/
		class tensor : public io::tensor 
		{
			public:
				tensor():io::tensor(){
					this->dtype=CPU;
				}
				tensor(char*, char *);
				tensor(double *, size_t *, size_t, size_t);
				~tensor();


		};
		tensor bias(tensor, int, int);

	}	

	io::cpu::tensor cuda_to_cpu(io::cuda::tensor input)
	{
		double *data;
		data = (double *) std::malloc(input.size * sizeof(double));
		cudaMemcpy(data, input.data, input.size * sizeof(double), cudaMemcpyDeviceToHost);
		io::cpu::tensor result(data, input.shape, input.size, input.dims);
		free(data);
		return result;
	}

	io::cuda::tensor cpu_to_cuda(io::cpu::tensor input)
	{

		double *data;
		cudaMalloc(&data, (input.size) * sizeof(double));
		
		cudaError_t error = cudaGetLastError(); 
		if(error!=cudaSuccess)
		{ 
			fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error));
		}
		cudaMemcpy(data, input.data, (input.size) * sizeof(double), cudaMemcpyHostToDevice);
		io::cuda::tensor result(data, input.shape, input.size, input.dims);
		cudaFree(data);
		return result;
	
	}


	template <class T>
	T one()
	{
		;
	};

	template <class T>
	T zeros()
	{
		;
	};

    template <class T>
    bool check(T *input1, T *input2);

	template <class C>
	io::cuda::tensor operator+(io::cuda::tensor, C);

	template <class C>
	io::cuda::tensor operator-(io::cuda::tensor, C);

	template <class C>
	io::cuda::tensor operator*(io::cuda::tensor, C);

	template <class C>
	io::cuda::tensor operator/(io::cuda::tensor, C);


	template <class C>
	io::cuda::tensor operator+(C, io::cuda::tensor);
	template <class C>
	io::cuda::tensor operator-(C, io::cuda::tensor);
	template <class C>
	io::cuda::tensor operator*(C, io::cuda::tensor);
	template <class C>
	io::cuda::tensor operator/(C, io::cuda::tensor);


    io::cuda::tensor operator+(io::cuda::tensor, io::cuda::tensor);

    io::cuda::tensor operator-(io::cuda::tensor, io::cuda::tensor);

    io::cuda::tensor operator*(io::cuda::tensor, io::cuda::tensor);

    io::cuda::tensor operator/(io::cuda::tensor, io::cuda::tensor);


    

	template <class C>
	io::cpu::tensor operator+(io::cpu::tensor, C);

	template <class C>
	io::cpu::tensor operator-(io::cpu::tensor, C);

	template <class C>
	io::cpu::tensor operator*(io::cpu::tensor, C);

	template <class C>
	io::cpu::tensor operator/(io::cpu::tensor, C);


	template <class C>
	io::cpu::tensor operator+(C, io::cpu::tensor);
	template <class C>
	io::cpu::tensor operator-(C, io::cpu::tensor);
	template <class C>
	io::cpu::tensor operator*(C, io::cpu::tensor);
	template <class C>
	io::cpu::tensor operator/(C, io::cpu::tensor);



    io::cpu::tensor operator+(io::cpu::tensor, io::cpu::tensor);

    io::cpu::tensor operator-(io::cpu::tensor, io::cpu::tensor);

    io::cpu::tensor operator*(io::cpu::tensor, io::cpu::tensor);

    io::cpu::tensor operator/(io::cpu::tensor, io::cpu::tensor);
	template <class T>
	T sin(T);


	template <class T>
	T cos(T);

}

template <class T>
bool io::check(T *input1, T *input2)
{
	if((*input1).size == (*input2).size)
	{
		return true;
	}
	return false;
}

io::cpu::tensor::~tensor(){
	;
}

io::cuda::tensor::~tensor(){
	;
}

io::cpu::tensor::tensor(char *filepath, char *varname)
{	
	data::get_size(&size, &dims, filepath, varname);
	data = (double *) std::malloc(size * sizeof(double));
	shape = (size_t *) std::malloc(dims * sizeof(size_t));
	data::read(data, shape, filepath, varname);


}

io::cpu::tensor::tensor
(
	double *input_data, 
	size_t *input_shape, 
	size_t input_size, 
	size_t input_dims
)
{
	size = input_size;
	dims = input_dims;
	data = (double *) std::malloc(size * sizeof(double));
	shape = (size_t *) std::malloc(dims * sizeof(size_t));
	memcpy(data, input_data, size * sizeof(double));
	memcpy(shape, input_shape, dims * sizeof(size_t));
	
}

io::cuda::tensor::tensor(char *filepath, char *varname)
{	
	/**
	 * I am learning how to build this part with RTX IO, so loading will will faster.
	 * 
	 */
	;

}


io::cuda::tensor::tensor
(
	double *input_data, 
	size_t *input_shape, 
	size_t input_size, 
	size_t input_dims
)
{
	size = input_size;
	dims = input_dims;

	shape = (size_t *) std::malloc(dims * sizeof(size_t));
	cudaMalloc(&data, size * sizeof(double));
	cudaMemcpy(data, input_data, size * sizeof(double), cudaMemcpyDeviceToDevice);
	memcpy(shape, input_shape, dims * sizeof(size_t));
	
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
__global__ void _3da(double *input1, double *input2, double *result, int d1, int d2, int d3, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int z_index = blockIdx.z * blockDim.z + threadIdx.z;	

	int index = x_index + y_index * d1 + z_index * d1 * d2;
	if(index < d1 * d2 * d3)
		result[index] = (*func)(input1[index], input2[index]);
}

__global__ void _3da(double *input1, double input2, double *result, int d1, int d2, int d3, double (*func)(double, double))
{
	// CONSTANT OPTIMIZATION
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int z_index = blockIdx.z * blockDim.z + threadIdx.z;	

	int index = x_index + y_index * d1 + z_index * d1 * d2;
	if(index < d1 * d2 * d3)
		result[index] = (*func)(input1[index], input2);
}


// 二维代数计算
__global__ void _2da(double *input1, double *input2, double *result, int d1, int d2, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * d1;
	if(index < d1 * d2)
		result[index] = (*func)(input1[index], input2[index]);
}

__global__ void _2da(double *input1, double input2, double *result, int d1, int d2, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * d1;
	if(index < d1 * d2)
		result[index] = (*func)(input1[index], input2);
}

// 代数计算
__global__ void _1da(double *input1, double *input2, double *result, int d1, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = x_index;
	if(index < d1)
		result[index] = (*func)(input1[index], input2[index]);
}


__global__ void _1da(double *input1, double input2, double *result, int d1, double (*func)(double, double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = x_index;
	if(index < d1)
		result[index] = (*func)(input1[index], input2);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void _3da(double *input, double *result, int d1, int d2, int d3, double (*func)(double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int z_index = blockIdx.z * blockDim.z + threadIdx.z;	

	int index = x_index + y_index * d1 + z_index * d1 * d2;
	if(index < d1 * d2 * d3)
		result[index] = (*func)(input[index]);
}


__global__ void _2da(double *input, double *result, int d1, int d2, double (*func)(double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;	

	int index = x_index + y_index * d1;
	if(index < d1 * d2)
		result[index] = (*func)(input[index]);
}


__global__ void _1da(double *input, double *result, int d1, double (*func)(double))
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = x_index;
	if(index < d1)
		result[index] = (*func)(input[index]);
}


int
kernel(double *input1, double input2, double *result, size_t d1, double (*func)(double, double))
{
	_1da<<<dim3((d1 + threadsPerBlock - 1) / threadsPerBlock), dim3(threadsPerBlock)>>>(input1, input2, result, d1, func);
	cudaDeviceSynchronize();
}

int
kernel(double *input1, double *input2, double *result, size_t d1, double (*func)(double, double))
{
	_1da<<<dim3((d1 + threadsPerBlock - 1) / threadsPerBlock), dim3(threadsPerBlock)>>>(input1, input2, result, d1, func);
	cudaDeviceSynchronize();
}


int
kernel(double *input1, double input2, double *result, size_t d1, size_t d2, double (*func)(double, double))
{
	_2da<<<dim3((d1 + threadsPerBlock - 1) / threadsPerBlock, (d2 + threadsPerBlock - 1) / threadsPerBlock), dim3(threadsPerBlock, threadsPerBlock)>>>(input1, input2, result, d1, d2, func);
	cudaDeviceSynchronize();		
}


int
kernel(double *input1, double *input2, double *result, size_t d1, size_t d2, double (*func)(double, double))
{
	_2da<<<dim3((d1 + threadsPerBlock - 1) / threadsPerBlock, (d2 + threadsPerBlock - 1) / threadsPerBlock), dim3(threadsPerBlock, threadsPerBlock)>>>(input1, input2, result, d1, d2, func);
	cudaDeviceSynchronize();		
}

int 
kernel(double *input1, double input2, double *result, size_t d1, size_t d2, size_t d3, double (*func)(double, double))
{
	_3da<<<dim3((d1 + threadsPerBlock - 1) / threadsPerBlock, (d2 + threadsPerBlock - 1) / threadsPerBlock,(d3 + threadsPerBlock - 1) / threadsPerBlock), dim3(threadsPerBlock, threadsPerBlock, threadsPerBlock)>>>(input1, input2, result, d1, d2, d3, func);
	cudaDeviceSynchronize();	
}


int 
kernel(double *input1, double *input2, double *result, size_t d1, size_t d2, size_t d3, double (*func)(double, double))
{
	_3da<<<dim3((d1 + threadsPerBlock - 1) / threadsPerBlock, (d2 + threadsPerBlock - 1) / threadsPerBlock,(d3 + threadsPerBlock - 1) / threadsPerBlock), dim3(threadsPerBlock, threadsPerBlock, threadsPerBlock)>>>(input1, input2, result, d1, d2, d3, func);
	cudaDeviceSynchronize();	
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
kernel(double *input, double *result, size_t d1, double (*func)(double))
{
	_1da<<<dim3(d1 / threadsPerBlock + 1), dim3(threadsPerBlock)>>>(input, result, d1, func);
	cudaDeviceSynchronize();
}


int
kernel(double *input, double *result, size_t d1, size_t d2, double (*func)(double))
{
	_2da<<<dim3(d1 / threadsPerBlock + 1, d2 / threadsPerBlock + 1), dim3(threadsPerBlock, threadsPerBlock)>>>(input, result, d1, d2, func);
	cudaDeviceSynchronize();		
}



int 
kernel(double *input, double *result, size_t d1, size_t d2, size_t d3, double (*func)(double))
{
	_3da<<<dim3(d1 / threadsPerBlock + 1, d2 / threadsPerBlock + 1, d3 / threadsPerBlock + 1), dim3(threadsPerBlock, threadsPerBlock, threadsPerBlock)>>>(input, result, d1, d2, d3, func);
	cudaDeviceSynchronize();	
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int 
broadcast
(
	double *input1, 
	double *input2, 
	double *result, 
	size_t dims, 
	size_t size,
	size_t* shape,
	bool is_constant,
	double (*func)(double, double)
)
{
	/*
	* This funtion only supports up to 3dim gpu array, but please notice most gpu only supports 3d array either
	* If you want to implement the function to higher dim array, I think there should be a higher block to call this function.
	*/
	size_t d1 = shape[0];
	size_t d2;
	size_t d3;
	

	
	switch(dims){
		case 1:
			{
				if (is_constant)
				{
					double *input2_gpu;
					cudaMalloc(&input2_gpu, sizeof(double));
					cudaMemcpy(input2_gpu, input2, sizeof(double), cudaMemcpyHostToDevice);
					kernel(input1, input2_gpu[0], result, d1, func);
					cudaFree(input2_gpu);
				}
				else
				{
					kernel(input1, input2, result, d1, func);
				}
				break;
			}
		case 2:
			{
				d2 = shape[1];
				if (is_constant)
				{
					double *input2_gpu;
					cudaMalloc(&input2_gpu, sizeof(double));
					cudaMemcpy(input2_gpu, input2, sizeof(double), cudaMemcpyHostToDevice);

					kernel(input1, input2_gpu, result, d1, d2, func);
					cudaFree(input2_gpu);
				}
				else
				{
					kernel(input1, input2, result, d1, d2, func);
				}		
				break;
			}
		case 3:
			{
				d2 = shape[1];
				d3 = shape[2];
				if (is_constant)
				{
					double *input2_gpu;
					cudaMalloc(&input2_gpu, sizeof(double));
					cudaMemcpy(input2_gpu, input2, sizeof(double), cudaMemcpyHostToDevice);
					kernel(input1, input2_gpu[0], result, d1, d2, d3, func);
					cudaFree(input2_gpu);
				}
				else
				{
					kernel(input1, input2, result, d1, d2, d3, func);
				}
				break;
			}
	}
	return EXIT_SUCCESS;
}


int 
broadcast
(
	double *input, 
	double *result, 
	size_t dims, 
	size_t size,
	size_t* shape,
	double (*func)(double)
)
{
	/*
	* This funtion only supports up to 3dim gpu array, but please notice most gpu only supports 3d array either
	* If you want to implement the function to higher dim array, I think there should be a higher block to call this function.
	*/
	size_t d1 = shape[0];
	size_t d2;
	size_t d3;
	

	
	switch(dims){
		case 1:
			{
				kernel(input, result, d1, func);
				break;
			}
		case 2:
			{
				d2 = shape[1];
				kernel(input, result, d1, d2, func);
				break;
			}
		case 3:
			{
				d2 = shape[1];
				d3 = shape[2];
				kernel(input, result, d1, d2, d3, func);
				break;				
			}
	}
	return EXIT_SUCCESS;
}


io::cuda::tensor 
io::operator+(io::cuda::tensor base, io::cuda::tensor element)
{
    if(io::check(&base, &element))
    {
        // Check funciton runs a size and shape check
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_add, sizeof(FP));
        broadcast(base.data, element.data, result_data, base.dims, base.size, base.shape, 0, fp_h);
		io::cuda::tensor result(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
		return result;
    }
	else
	{
		throw std::runtime_error(
			"Tensor shape mismatched!"
		);
	}
}


template <class C>
io::cuda::tensor 
io::operator+(io::cuda::tensor base, C element)
{
    try
	{
        // Check funciton runs a size and shape check
		double element_data = double(element);
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_add, sizeof(FP));
        broadcast(base.data, &element_data, result_data, base.dims, base.size, base.shape, 1, fp_h);

		io::cuda::tensor result(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
}


template <class C>
io::cuda::tensor 
io::operator+(C element, io::cuda::tensor base)
{
    try
	{
        // Check funciton runs a size and shape check
		double element_data = double(element);
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_add, sizeof(FP));
        broadcast(base.data, &element_data, result_data, base.dims, base.size, base.shape, 1, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
}


io::cuda::tensor 
io::operator-(io::cuda::tensor base, io::cuda::tensor element)
{
    if(io::check(&base, &element))
    {
        // Check funciton runs a size and shape check
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_sub, sizeof(FP));
        broadcast(base.data, element.data, result_data, base.dims, base.size, base.shape, 0, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
		
    }
	else
	{
		throw std::runtime_error(
			"Tensor shape mismatched!"
		);
	}
}

template <class C>
io::cuda::tensor 
io::operator-(io::cuda::tensor base, C element)
{
    try
	{
        // Check funciton runs a size and shape check
		double element_data = double(element);
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_sub, sizeof(FP));
        broadcast(base.data, &element_data, result_data, base.dims, base.size, base.shape, 1, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
}

template <class C>
io::cuda::tensor 
io::operator-(C element, io::cuda::tensor base)
{
    try
	{
        // Check funciton runs a size and shape check
		double element_data = double(element);
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_sub, sizeof(FP));
        broadcast(base.data, &element_data, result_data, base.dims, base.size, base.shape, 1, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
}

io::cuda::tensor 
io::operator*(io::cuda::tensor base, io::cuda::tensor element)
{
    if(io::check(&base, &element))
    {
        // Check funciton runs a size and shape check
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_mul, sizeof(FP));
        broadcast(base.data, element.data, result_data, base.dims, base.size, base.shape, 0, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
		
    }
	else
	{
		throw std::runtime_error(
			"Tensor shape mismatched!"
		);
	}
}

template <class C>
io::cuda::tensor 
io::operator*(io::cuda::tensor base, C element)
{
    try
	{
        // Check funciton runs a size and shape check
		double element_data = double(element);
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_mul, sizeof(FP));
        broadcast(base.data, &element_data, result_data, base.dims, base.size, base.shape, 1, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
}


template <class C>
io::cuda::tensor 
io::operator*(C element, io::cuda::tensor base)
{
    try
	{
        // Check funciton runs a size and shape check
		double element_data = double(element);
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_mul, sizeof(FP));
        broadcast(base.data, &element_data, result_data, base.dims, base.size, base.shape, 1, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
}

io::cuda::tensor 
io::operator/(io::cuda::tensor base, io::cuda::tensor element)
{
    if(io::check(&base, &element))
    {
        // Check funciton runs a size and shape check
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_div, sizeof(FP));
        broadcast(base.data, element.data, result_data, base.dims, base.size, base.shape, 0, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
		
    }
	else
	{
		throw std::runtime_error(
			"Tensor shape mismatched!"
		);
	}
}

template <class C>
io::cuda::tensor 
io::operator/(io::cuda::tensor base, C element)
{
    try
	{
        // Check funciton runs a size and shape check
		double element_data = double(element);
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_div, sizeof(FP));
        broadcast(base.data, &element_data, result_data, base.dims, base.size, base.shape, 1, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
}

template <class C>
io::cuda::tensor 
io::operator/(C element, io::cuda::tensor base)
{
    try
	{
        // Check funciton runs a size and shape check
		double element_data = double(element);
        double *result_data;
    
		cudaMalloc(&result_data, base.size * sizeof(double));
		FP fp_h;
		cudaMemcpyFromSymbol(&fp_h, fp_div, sizeof(FP));
        broadcast(base.data, &element_data, result_data, base.dims, base.size, base.shape, 1, fp_h);

		io::cuda::tensor result = io::cuda::tensor(result_data, base.shape, base.size, base.dims);
		cudaFree(result_data);
        return result;
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
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


io::cuda::tensor io::cuda::bias(io::cuda::tensor base, int i, int j)
{
	/*
	* 该函数只针对2d数组进行定义，但是不检查数组形状
	*/

	double *result_i;
	double *result_j;

	cudaMalloc(&result_i, base.size * sizeof(double));
	cudaMalloc(&result_j, base.size * sizeof(double));


	size_t d1 = base.shape[0];
	size_t d2 = base.shape[1];
	
	FP fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_add, sizeof(FP));
	bias_i<<<dim3(d1 / threadsPerBlock + 1, d2 / threadsPerBlock + 1), dim3(threadsPerBlock, threadsPerBlock)>>>(base.data, result_i, d1, d2, i);
	bias_j<<<dim3(d1 / threadsPerBlock + 1, d2 / threadsPerBlock + 1), dim3(threadsPerBlock, threadsPerBlock)>>>(result_i, result_j, d1, d2, j);
	//f2d<<<dim3(d1 / threadsPerBlock + 1, d2 / threadsPerBlock + 1), dim3(threadsPerBlock, threadsPerBlock)>>>(gresult_i, gresult_j, gresult, d1, d2, fp_h);
	
	io::cuda::tensor output=io::cuda::tensor(result_j, base.shape, base.size, base.dims);
	
	cudaFree(result_i);
	cudaFree(result_j);
	return output;
}



typedef double (*FP1var)(double);
__device__ double dsin(double x) {return sin(x);}
__device__ double dcos(double x) {return cos(x);}
__device__ FP1var fp_sin = dsin;
__device__ FP1var fp_cos = dcos;

template <class T=io::cuda::tensor>
T sin(T _this)
{

	double *result;
	cudaMalloc(&result, _this.size * sizeof(double));
	FP1var fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_sin, sizeof(FP));
	broadcast(_this.data, result, _this.dims, _this.size, _this.shape, fp_h);

	T output = tensor(result, _this.shape, _this.size, _this.dims);
	cudaFree(result);
	return output;
}

template <class T=io::cuda::tensor>
T cos(T _this)
{

	double *result;
	cudaMalloc(&result, _this.size * sizeof(double));
	FP1var fp_h;
	cudaMemcpyFromSymbol(&fp_h, fp_cos, sizeof(FP));
	broadcast(_this.data, result, _this.dims, _this.size, _this.shape, fp_h);

	T output = tensor(result, _this.shape, _this.size, _this.dims);
	cudaFree(result);
	return output;
}

