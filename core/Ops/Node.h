#include <functional>
#include <map>
#include <vector>
#include <core/function/Prop.h>
#include <string>
#include <core/variables/Variables.h>
/*
 * Ops 类的作用是建立 Variable类和以double* 等基础类型为接口的函数之间的桥梁
*/

/* 矩阵之间四则运算的参数列表
 * F1 : double* x, double* y, double* result, int N
 * F2 : double* x, double* y, double* result, int N1, int N2
 * F3 : double* x, double* y, double* z, double* result, int N1, int N2, int N3
*/
using F1 = int(double*, double*, double*, int);
using F2 = int(double*, double*, double*, int, int);
using F3 = int(double*, double*, double*, int, int, int);

/*
 * 矩阵与常数之间四则运算的参数列表
 * F1 : double* x, double y, double* result, int N
 * F2 : double* x, double y, double* result, int N1, int N2
 * F3 : double* x, double y, double* result, int N1, int N2, int N3 
*/
using F1_c = int(double*, double, double*, int);
using F2_c = int(double*, double, double*, int, int);
using F3_c = int(double*, double, double*, int, int, int);

/*
 * 数学函数
 * double* input, double* result, int N
 * double* input, double* result, int N1, int N2 
 * double* input, double* result, int N1, int N2, int N3  
*/
using M1 = int(double*, double*, int);
using M2 = int(double*, double*, int, int);
using M3 = int(double*, double*, int, int, int);


/*
 * 初始化函数
 * double* arr, int N
 * double* input, int N1, int N2
 * double* input, int N1, int N2, int N3
*/
using I1 = int(double*, double, int);
using I2 = int(double*, double, int, int);
using I3 = int(double*, double, int, int, int);



template <class Ftype>
class Ops
{
    private:
        std::function<Ftype> func;
        std::Variable func;
        Prop::shape s_;
        std::vector<Ops*> node_list;
    public:
        Ops::Ops(Ftype f);
        int register_input_node(Ops* node);
        int run__();
};


template<class Ftype>
Ops::Ops(Ftype f)
{
    func = f;
}


template<class Ftype>
int Ops<Ftype>::register_input_node(Ops* node)
{
    node_list.push_back(node);
}