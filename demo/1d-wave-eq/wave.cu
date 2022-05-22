// wave equation solver


#include <tensor.h>
#include <iostream>

int main(void)
{
    int c{1};
    int xmin{0}, xmax{10}, Nx{10};
    int tmin{0}, tmax{15}, Nt{20};

    /*
    *  这里需要一个linspace
    */
    io::cpu::tensor xc = io::cpu::linspace(xmin, xmax, Nx);
    io::cpu::tensor tc = io::cpu::linspace(tmin, tmax, Nt);

    /*
    * GPU 化
    */
    
    io::cuda::tensor x = io::cpu_to_cuda(xc);
    io::cuda::tensor t = io::cpu_to_cuda(tc);
    double hx = double(xmax - xmin) / double(Nx - 1);
    double ht = double(tmax - tmin) / double(Nt - 1);
    size_t u_shape[2] = {Nx, Nt};
    io::cpu::tensor uc=io::zeros(u_shape);
    io::cuda::tensor u=io::cpu_to_cuda(uc);
    /*
    for(int j=1; j<Nt-1; j++)
    {
        if(tc.data[j] < 5)
        {
            
        }
        else
        {

        }
    }
    */
    
}