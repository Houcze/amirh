#include <core/mem.h>
#include <iostream>
#include <cuda_runtime.h>

double *Nallocator::register_variable(std::string vname)
{
    double *v;
    cudaMalloc(&v, Prop::size(NOSH) * sizeof(double));
    vmap[vname] = v;
    vargs[vname] = NOSH;
    return v;
}

double *Nallocator::register_variable(std::string vname, Prop::shape s)
{
    double *v;
    cudaMalloc(&v, sizeof(double) * Prop::size(s));
    vmap[vname] = v;
    vargs[vname] = s;
    return v;
}

void Nallocator::print_vmap_length()
{
    std::cout << vmap.size() << std::endl;
}

double *Nallocator::require_variable(std::string vname)
{
    return vmap[vname];
}

bool Nallocator::isn(std::string vname)
{
    return vmap.count(vname);
}

int Nallocator::deallocate_variable(double *v, std::string vname)
{
    cudaFree(v);
    vmap.erase(vmap.find(vname));
    return EXIT_SUCCESS;
}

int Nallocator::register_shape(Prop::shape s_)
{
    NOSH = s_;
    return EXIT_SUCCESS;
}

int Nallocator::register_seqlen(int len)
{
    for (int i = 0; i < len; i++)
    {
        register_variable("v" + std::to_string(i));
    }
    return EXIT_SUCCESS;
}

void Nallocator::print_variable_list()
{
    for (auto it : vmap)
    {
        std::cout << it.first << std::endl;
    }
}

Prop::shape Nallocator::registered_shape()
{
    return NOSH;
}
