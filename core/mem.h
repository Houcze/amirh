#ifndef MEM_H
#define MEM_H
#include <map>
#include <core/function/Prop.h>
#include <core/function/func.h>
#include <string>
#include <tuple>
#include <iostream>


/**
 * @brief 注意，Nallocator类所分配的变量的shape未必一致，但是也可以全部一致
 */
class Nallocator
{
private:
    Prop::shape NOSH;
    std::map<std::string, double *> vmap;
    std::map<std::string, Prop::shape> vargs;
public:
    // 注册标准队列长度
    int register_shape(Prop::shape);
    // 注册预设队列长度
    int register_seqlen(int);
    int deallocate_variable(double *, std::string);

    double *register_variable(std::string);
    double *require_variable(std::string);
    
    double *register_variable(std::string, Prop::shape);

    bool isn(std::string);
    Prop::shape registered_shape();
    void print_variable_list();
    void print_vmap_length();
};

extern Nallocator Npool;

#endif
