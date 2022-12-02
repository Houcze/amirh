#ifndef MEM_H
#define MEM_H
#include <map>
#include <core/function/Prop.h>
#include <core/function/func.h>
#include <string>
#include <tuple>
#include <iostream>

class Nallocator
{
private:
    Prop::shape NOSH;
    std::map<std::string, double *> vmap;

public:
    int register_shape(Prop::shape);
    int register_seqlen(int);
    int deallocate_variable(double *, std::string);
    double *register_variable(std::string);
    double *require_variable(std::string);
    bool isn(std::string);
    void print_variable_list();
};

extern Nallocator Npool;

#endif
