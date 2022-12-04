#include <ctime>
#include <io/netcdf>
#include <core/mem.h>
#include <cuda_runtime.h>
#include <core/function/Prop.h>
#include <core/variables/Variables.h>

Nallocator Npool;

int Variable::operator=(Variable a)
{
    shape = a.shape;
    val = Npool.require_variable(name);

    if (val != a.val)
    {
        cudaMemcpy(val, a.val, Prop::size(shape) * sizeof(double), cudaMemcpyDeviceToDevice);
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
    if (Npool.isn(name))
    {
        val = Npool.require_variable(name);
    }
    else
    {
        val = Npool.register_variable(name);
    }
    val = v;
}

Variable::Variable(const Variable &a)
{
    name = a.name;
    shape = a.shape;
    if (Npool.isn(name))
    {
        val = Npool.require_variable(name);
    }
    else
    {
        val = Npool.register_variable(name);
    }
    val = a.val;
}

Variable operator+(const Variable &a, const Variable &b)
{

    double *result;
    std::string vname;
    if (Npool.isn(a.name + "+" + b.name))
    {
        result = Npool.require_variable(a.name + "+" + b.name);
        vname = a.name + "+" + b.name;
    }
    else if (Npool.isn(b.name + "+" + a.name))
    {
        result = Npool.require_variable(b.name + "+" + a.name);
        vname = b.name + "+" + a.name;
    }
    else
    {
        result = Npool.register_variable(a.name + "+" + b.name);
        vname = a.name + "+" + b.name;
    }

    Prop::shape s0 = a.shape;
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
    std::string vname;
    if (Npool.isn(a.name + "+" + std::to_string(b)))
    {
        result = Npool.require_variable(a.name + "+" + std::to_string(b));
        vname = a.name + "+" + std::to_string(b);
    }
    else if (Npool.isn(std::to_string(b) + "+" + a.name))
    {
        result = Npool.require_variable(std::to_string(b) + "+" + a.name);
        vname = std::to_string(b) + "+" + a.name;
    }
    else
    {
        result = Npool.register_variable(a.name + "+" + std::to_string(b));
        vname = a.name + "+" + std::to_string(b);
    }

    Prop::shape s0 = a.shape;
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
    std::string vname;
    if (Npool.isn(std::to_string(a) + "+" + b.name))
    {
        result = Npool.require_variable(std::to_string(a) + "+" + b.name);
        vname = std::to_string(a) + "+" + b.name;
    }
    else if (Npool.isn(b.name + "+" + std::to_string(a)))
    {
        result = Npool.require_variable(b.name + "+" + std::to_string(a));
        vname = b.name + "+" + std::to_string(a);
    }
    else
    {
        result = Npool.register_variable(std::to_string(a) + "+" + b.name);
        vname = std::to_string(a) + "+" + b.name;
    }

    Prop::shape s0 = b.shape;
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
    std::string vname;
    if (Npool.isn(a.name + "*" + b.name))
    {
        result = Npool.require_variable(a.name + "*" + b.name);
        vname = a.name + "*" + b.name;
    }
    else if (Npool.isn(b.name + "*" + a.name))
    {
        result = Npool.require_variable(b.name + "*" + a.name);
        vname = b.name + "*" + a.name;
    }
    else
    {
        result = Npool.register_variable(a.name + "*" + b.name);
        vname = a.name + "*" + b.name;
    }
    Prop::shape s0 = a.shape;
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
    std::string vname;
    if (Npool.isn(a.name + "*" + std::to_string(b)))
    {
        result = Npool.require_variable(a.name + "*" + std::to_string(b));
        vname = a.name + "*" + std::to_string(b);
    }
    else if (Npool.isn(std::to_string(b) + "*" + a.name))
    {
        result = Npool.require_variable(std::to_string(b) + "*" + a.name);
        vname = std::to_string(b) + "*" + a.name;
    }
    else
    {
        result = Npool.register_variable(a.name + "*" + std::to_string(b));
        vname = a.name + "*" + std::to_string(b);
    }
    Prop::shape s0 = a.shape;
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
    std::string vname;
    if (Npool.isn(std::to_string(a) + "*" + b.name))
    {
        result = Npool.require_variable(std::to_string(a) + "*" + b.name);
        vname = std::to_string(a) + "*" + b.name;
    }
    else if (Npool.isn(b.name + "*" + std::to_string(a)))
    {
        result = Npool.require_variable(b.name + "*" + std::to_string(a));
        vname = b.name + "*" + std::to_string(a);
    }
    else
    {
        result = Npool.register_variable(std::to_string(a) + "*" + b.name);
        vname = std::to_string(a) + "*" + b.name;
    }

    Prop::shape s0 = b.shape;
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
    if (Npool.isn(a.name + "-" + b.name))
    {
        result = Npool.require_variable(a.name + "-" + b.name);
    }
    else
    {
        result = Npool.register_variable(a.name + "-" + b.name);
    }
    Prop::shape s0 = a.shape;
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
    if (Npool.isn(a.name + "-" + std::to_string(b)))
    {
        result = Npool.require_variable(a.name + "-" + std::to_string(b));
    }
    else
    {
        result = Npool.register_variable(a.name + "-" + std::to_string(b));
    }
    Prop::shape s0 = a.shape;
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
    if (Npool.isn(std::to_string(a) + "-" + b.name))
    {
        result = Npool.require_variable(std::to_string(a) + "-" + b.name);
    }
    else
    {
        result = Npool.register_variable(std::to_string(a) + "-" + b.name);
    }
    Prop::shape s0 = b.shape;
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
    if (Npool.isn(a.name + "/" + b.name))
    {
        result = Npool.require_variable(a.name + "/" + b.name);
    }
    else
    {
        result = Npool.register_variable(a.name + "/" + b.name);
    }
    Prop::shape s0 = a.shape;
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
    if (Npool.isn(a.name + "/" + std::to_string(b)))
    {
        result = Npool.require_variable(a.name + "/" + std::to_string(b));
    }
    else
    {
        result = Npool.register_variable(a.name + "/" + std::to_string(b));
    }
    Prop::shape s0 = a.shape;
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
    if (Npool.isn(std::to_string(a) + "/" + b.name))
    {
        result = Npool.require_variable(std::to_string(a) + "/" + b.name);
    }
    else
    {
        result = Npool.register_variable(std::to_string(a) + "/" + b.name);
    }
    Prop::shape s0 = b.shape;
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
    if (Npool.isn("sin" + a.name))
    {
        result = Npool.require_variable("sin" + a.name);
    }
    else
    {
        result = Npool.register_variable("sin" + a.name);
    }
    Prop::shape s0 = a.shape;
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
    if (Npool.isn("cos" + a.name))
    {
        result = Npool.require_variable("cos" + a.name);
    }
    else
    {
        result = Npool.register_variable("cos" + a.name);
    }
    Prop::shape s0 = a.shape;
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
    if (Npool.isn("tan" + a.name))
    {
        result = Npool.require_variable("tan" + a.name);
    }
    else
    {
        result = Npool.register_variable("tan" + a.name);
    }
    Prop::shape s0 = a.shape;
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
    if (Npool.isn(a.name + "(i)" + std::to_string(i) + "(j)" + std::to_string(j)))
    {
        result = Npool.require_variable(a.name + "(i)" + std::to_string(i) + "(j)" + std::to_string(j));
    }
    else
    {
        result = Npool.register_variable(a.name + "(i)" + std::to_string(i) + "(j)" + std::to_string(j));
    }
    Prop::shape s0 = a.shape;
    parti_diff(a.val, result, s0["d1"], s0["d2"], i, j);
    return Variable(a.name + "(i)" + std::to_string(i) + "(j)" + std::to_string(j), s0, result);
}

Variable zero_boundary(const Variable &a)
{
    Prop::shape s0 = a.shape;
    zero_boundary(a.val, s0["d1"], s0["d2"]);
    return a;
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
    s["d1"] = p["latitude"];
    s["d2"] = p["longitude"];

    std::string vname = std::string(name);
    // 此处没有检查变量是否存在于变量池
    double *host_val;
    host_val = (double *) malloc(Prop::size(s) * sizeof(double));
    double *val;
    if(!Npool.isn(vname))
    {
        val = Npool.register_variable(vname);
    }
    else
    {
        std::cout << std::string(name) + " has already been registered!" << std::endl;
        vname = std::string(name) + "_" + std::to_string(std::time(0));
        std::cout << vname + " will be registered!" << std::endl;
        val = Npool.register_variable(vname);
    }
    netcdf::ds(host_val, filepath, name);
    cudaMemcpy(val, host_val, Prop::size(s) * sizeof(double), cudaMemcpyHostToDevice);
    free(host_val);
    return Variable(vname, s, val);
}




