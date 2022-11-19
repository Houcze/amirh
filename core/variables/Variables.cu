#include <core/function/Prop.h>
#include <cuda_runtime.h>
#include <string>
#include <core/variables/Variables.h>


std::Variable::Variable(Prop::shape s, std::Name n)
{
    s_ = s;
    cudaMalloc(&ds, Prop::size(s_) * sizeof(double));
    cudaMemset(ds, 0, Prop::size(s_) * sizeof(double));
    vname = n;    
}

std::Variable::Variable(Prop::shape s)
{
    s_ = s;
    cudaMalloc(&ds, Prop::size(s_) * sizeof(double));
    cudaMemset(ds, 0, Prop::size(s_) * sizeof(double));   
}


int std::Variable::register_name(std::Name name)
{
    vname = name;
}


std::Variable::~Variable()
{
    cudaFree(ds);
}


int std::Variable::size()
{
    return Prop::size(s_);
}


int std::Variable::dims()
{
    return Prop::dims(s_);
}

