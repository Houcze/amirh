#pragma once

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
			~tensor();
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
				double *get_data();

		};

	}	

	io::cpu::tensor cpu_to_cuda(io::cpu::tensor input)
	{

		double *data_cuda;
		data_cuda = (double *) std::malloc(input.size * sizeof(double));
		std::memcpy(data_cuda, input.data, input.size * sizeof(double));
		free(input.data);
		cudaMalloc(&(input.data), input.size * sizeof(double));
		cudaMemcpy(input.data, data_cuda, input.size * sizeof(double), cudaMemcpyHostToDevice);
		free(data_cuda);
		input.dtype = CUDA;
	
	}

	io::cuda::tensor cuda_to_cpu(io::cuda::tensor input)
	{
		double *data_cpu;
		data_cpu = (double *) std::malloc(input.size * sizeof(double));
		cudaMemcpy(data_cpu, input.data, input.size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(input.data);
		input.data = (double *) std::malloc(input.size * sizeof(double));
		std::memcpy(input.data, data_cpu, input.size * sizeof(double));	
		input.dtype = CPU;
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

    io::cpu::tensor operator+(io::cpu::tensor, io::cpu::tensor);

    io::cpu::tensor operator-(io::cpu::tensor, io::cpu::tensor);

    io::cpu::tensor operator*(io::cpu::tensor, io::cpu::tensor);

    io::cpu::tensor operator/(io::cpu::tensor, io::cpu::tensor);

    

	template <class T, class C>
	io::cpu::tensor operator+(T, C);

	template <class T, class C>
	io::cpu::tensor operator-(T, C);
		
	template <class T, class C>
	io::cpu::tensor operator*(T, C);

	template <class T, class C>
	io::cpu::tensor operator/(T, C);

	template <class T>
	T sin(T);


	template <class T>
	T cos(T);

}