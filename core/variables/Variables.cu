#include <ctime>
#include <any>
#include <io/netcdf>
#include <core/mem.h>
#include <cuda_runtime.h>
#include <core/function/Prop.h>
#include <core/variables/Variables.h>

Nallocator Npool;

/**
 * @brief 这个函数适用于标准长度被注册的场景
 * @param name
 * @param allocator
 * @return double*
 */
double *alloc(std::string name, Nallocator allocator = Npool)
{
    double *val;
    if (allocator.isn(name))
    {
        val = allocator.require_variable(name);
    }
    else
    {
        val = allocator.register_variable(name);
    }
    return val;
}

/**
 * @brief 这个函数适用于标准长度被注册的场景
 * @param name
 * @param ename
 * @param allocator
 * @return double*
 */
double *ealloc(std::string name, std::string ename, Nallocator allocator = Npool)
{
    double *val;
    if (!allocator.isn(name))
    {
        val = allocator.register_variable(name);
    }
    else
    {
        std::cout << name + " has already been registered!" << std::endl;
        std::cout << ename + " will be registered!" << std::endl;
        val = allocator.register_variable(ename);
    }
    return val;
}

double *alloc(std::string name, Prop::shape s, Nallocator &allocator = Npool)
{
    double *val;
    if (allocator.isn(name))
    {
        val = allocator.require_variable(name);
    }
    else
    {
        val = allocator.register_variable(name, s);
    }
    return val;
}

Variable ealloc(std::string name, std::string ename, Prop::shape s, Nallocator &allocator = Npool)
{
    double *val;
    std::string registered_name;
    if (allocator.isn(name))
    {
        std::cout << name + " has already been registered!" << std::endl;
        std::cout << ename + " will be registered!" << std::endl;
        val = allocator.register_variable(ename, s);
        registered_name = ename;
    }
    else
    {
        val = allocator.register_variable(name, s);
        registered_name = name;
    }
    return Variable(registered_name, s, val);
}

int Variable::operator=(Variable a)
{
    if ((shape == a.shape) || (Prop::dims(shape) == 0))
    {
        val = alloc(name, a.shape);
        if (val != a.val)
        {
            cudaMemcpy(val, a.val, Prop::size(a.shape) * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }
    if ((shape != a.shape) && (Prop::dims(shape) != 0))
    {
        // 此处将会注册一个同名但是形状不一致的变量
        val = alloc(name + Prop::to_string(a.shape), a.shape);
        if (val != a.val)
        {
            cudaMemcpy(val, a.val, Prop::size(a.shape) * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }
    return EXIT_SUCCESS;
}

Variable make_variable(std::string n, Prop::shape s, double *v)
{
    return Variable(n, s, v);
}

Variable::Variable(std::string n, Prop::shape s, double *v)
{
    name = n;
    shape = s;
    val = alloc(name, shape);
    val = v;
}

Variable::Variable(const Variable &a)
{
    name = a.name;
    shape = a.shape;
    val = alloc(name, shape);
    val = a.val;
}

Variable init_Variable(std::string name, Prop::shape s, const void *v)
{
    double *val;
    val = alloc(name, s);
    cudaMemcpy(val, v, sizeof(double) * Prop::size(s), cudaMemcpyHostToDevice);
    return Variable(name, s, val);
}

Variable operator+(const Variable &a, const Variable &b)
{
    double *result;
    std::string vname = a.name + "+" + b.name;
    Prop::shape s0 = a.shape;
    result = alloc(vname, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        add(a.val, b.val, result, s0["d1"]);
        break;
    case 2:
        add(a.val, b.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        add(a.val, b.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(vname, s0, result);
}

Variable operator+(const Variable &a, double b)
{

    double *result;
    Prop::shape s0 = a.shape;
    std::string vname = a.name + "+" + std::to_string(b);
    result = alloc(vname, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        add(a.val, b, result, s0["d1"]);
        break;
    case 2:
        add(a.val, b, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        add(a.val, b, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(vname, s0, result);
}

Variable operator+(double a, const Variable &b)
{

    double *result;
    Prop::shape s0 = b.shape;
    std::string vname = b.name + "+" + std::to_string(a);
    result = alloc(vname, s0);

    switch (Prop::dims(s0))
    {
    case 1:
        add(a, b.val, result, s0["d1"]);
        break;
    case 2:
        add(a, b.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        add(a, b.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(vname, s0, result);
}

Variable operator*(const Variable &a, const Variable &b)
{
    double *result;
    Prop::shape s0 = a.shape;

    std::string vname = a.name + "*" + b.name;
    result = alloc(vname, s0);

    switch (Prop::dims(s0))
    {
    case 1:
        mul(a.val, b.val, result, s0["d1"]);
        break;
    case 2:
        mul(a.val, b.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        mul(a.val, b.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(vname, s0, result);
}

Variable operator*(const Variable &a, double b)
{

    double *result;
    Prop::shape s0 = a.shape;
    std::string vname = std::to_string(b) + "*" + a.name;
    result = alloc(vname, s0);

    switch (Prop::dims(s0))
    {
    case 1:
        mul(a.val, b, result, s0["d1"]);
        break;
    case 2:
        mul(a.val, b, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        mul(a.val, b, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(vname, s0, result);
}

Variable operator*(double a, const Variable &b)
{

    double *result;
    Prop::shape s0 = b.shape;
    std::string vname = std::to_string(a) + "*" + b.name;
    result = alloc(vname, s0);

    switch (Prop::dims(s0))
    {
    case 1:
        mul(a, b.val, result, s0["d1"]);
        break;
    case 2:
        mul(a, b.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        mul(a, b.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(vname, s0, result);
}

Variable operator-(const Variable &a, const Variable &b)
{
    double *result;
    Prop::shape s0 = a.shape;
    result = alloc(a.name + "-" + b.name, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        sub(a.val, b.val, result, s0["d1"]);
        break;
    case 2:
        sub(a.val, b.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        sub(a.val, b.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(a.name + "-" + b.name, s0, result);
}

Variable operator-(const Variable &a, double b)
{
    double *result;
    Prop::shape s0 = a.shape;
    result = alloc(a.name + "-" + std::to_string(b), s0);
    switch (Prop::dims(s0))
    {
    case 1:
        sub(a.val, b, result, s0["d1"]);
        break;
    case 2:
        sub(a.val, b, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        sub(a.val, b, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(a.name + "-" + std::to_string(b), s0, result);
}

Variable operator-(double a, const Variable &b)
{
    double *result;
    Prop::shape s0 = b.shape;
    result = alloc(std::to_string(a) + "-" + b.name, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        sub(a, b.val, result, s0["d1"]);
        break;
    case 2:
        sub(a, b.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        sub(a, b.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(std::to_string(a) + "-" + b.name, s0, result);
}

Variable operator/(const Variable &a, const Variable &b)
{
    double *result;
    Prop::shape s0 = a.shape;
    result = alloc(a.name + "/" + b.name, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        div(a.val, b.val, result, s0["d1"]);
        break;
    case 2:
        div(a.val, b.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        div(a.val, b.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(a.name + "/" + b.name, s0, result);
}

Variable operator/(const Variable &a, double b)
{
    double *result;
    Prop::shape s0 = a.shape;
    result = alloc(a.name + "/" + std::to_string(b), s0);
    switch (Prop::dims(s0))
    {
    case 1:
        div(a.val, b, result, s0["d1"]);
        break;
    case 2:
        div(a.val, b, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        div(a.val, b, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(a.name + "/" + std::to_string(b), s0, result);
}

Variable operator/(double a, const Variable &b)
{
    double *result;
    Prop::shape s0 = b.shape;
    result = alloc(std::to_string(a) + "/" + b.name, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        div(a, b.val, result, s0["d1"]);
        break;
    case 2:
        div(a, b.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        div(a, b.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable(std::to_string(a) + "/" + b.name, s0, result);
}

Variable sin(const Variable &a)
{
    double *result;
    Prop::shape s0 = a.shape;
    result = alloc("sin" + a.name, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        sin(a.val, result, s0["d1"]);
        break;
    case 2:
        sin(a.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        sin(a.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable("sin" + a.name, s0, result);
}

Variable cos(const Variable &a)
{
    double *result;
    Prop::shape s0 = a.shape;
    result = alloc("cos" + a.name, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        cos(a.val, result, s0["d1"]);
        break;
    case 2:
        cos(a.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        cos(a.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable("cos" + a.name, s0, result);
}

Variable tan(const Variable &a)
{
    double *result;
    Prop::shape s0 = a.shape;
    result = alloc("tan" + a.name, s0);
    switch (Prop::dims(s0))
    {
    case 1:
        tan(a.val, result, s0["d1"]);
        break;
    case 2:
        tan(a.val, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        tan(a.val, result, s0["d1"], s0["d2"], s0["d3"]);
        break;
    default:
        break;
    }
    return Variable("tan" + a.name, s0, result);
}

Variable parti_diff(const Variable &a, int i, int j)
{
    double *result;
    Prop::shape s0 = a.shape;
    result = alloc(a.name + "(i)" + std::to_string(i) + "(j)" + std::to_string(j), s0);
    parti_diff(a.val, result, s0["d1"], s0["d2"], i, j);
    return Variable(a.name + "(i)" + std::to_string(i) + "(j)" + std::to_string(j), s0, result);
}

Variable laplace(const Variable &a)
{
    Variable result = parti_diff(a, 1, 0) + parti_diff(a, -1, 0) + parti_diff(a, 0, 1) + parti_diff(a, 0, -1) - 4 * a;
    return result;
}

int operator<<(Variable &a, const Variable &b)
{
    a.shape = b.shape;
    cudaMemcpy(a.val, b.val, Prop::size(a.shape) * sizeof(double), cudaMemcpyDeviceToDevice);
    return EXIT_SUCCESS;
}

Variable read_from_netcdf(char *filepath, char *name)
{
    Prop::shape p;
    netcdf::ds_prop(&p, filepath, name);
    Prop::shape s;
    // 这里没有错误检查是否可以赋值
    if (p.count("latitude"))
    {
        s["d1"] = p["latitude"];
    }
    else
    {
        s["d1"] = 1;
    }

    if (p.count("longitude"))
    {
        s["d2"] = p["longitude"];
    }
    else
    {
        s["d2"] = 1;
    }

    std::string vname = std::string(name);
    // 此处没有检查变量是否存在于变量池
    double *host_val;
    host_val = (double *)malloc(Prop::size(s) * sizeof(double));

    Variable result = ealloc(vname, std::string(name) + "_" + std::to_string(std::time(0)), s);
    netcdf::ds(host_val, filepath, name);
    cudaMemcpy(result.val, host_val, Prop::size(s) * sizeof(double), cudaMemcpyHostToDevice);
    free(host_val);
    return result;
}

Variable mul2(const Variable &a, const Variable &b)
{
    double *result;
    double *result_t;

    std::string vname = a.name + "." + b.name;
    Prop::shape s;
    Prop::shape s_t;
    Prop::shape s_a = a.shape;
    Prop::shape s_b = b.shape;
    s["d1"] = s_a["d1"];
    s["d2"] = s_b["d2"];
    s_t["d2"] = s["d1"];
    s_t["d1"] = s["d2"];

    int m = s_a["d1"];
    int n1 = s_a["d2"];
    int n2 = s_b["d1"];
    int n = (n1 + n2) / 2;
    int k = s_b["d2"];

    result_t = alloc(vname + ".t", s_t);
    result = alloc(vname, s);
    mmul(a.val, b.val, result_t, m, k, n);
    transpose(result_t, result, s["d2"], s["d1"]);
    return Variable(vname, s, result);
}

Variable ones2(Prop::shape s)
{
    double *val;
    std::string vname = "1d1:" + std::to_string(s["d1"]) + "d2:" + std::to_string(s["d2"]);
    val = alloc(vname, s);
    set_cons(val, 1., s["d1"], s["d2"]);
    return Variable(vname, s, val);
}

void print_info(const Variable &v)
{
    std::cout << v.name << " info:\n";
    Prop::print(v.shape);
}
