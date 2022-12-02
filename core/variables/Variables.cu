#include <core/mem.h>
#include <core/function/Prop.h>
#include <core/variables/Variables.h>

Nallocator Npool;

Variable operator+(Variable a, Variable b)
{

    double *result;
    std::string vname;
    if (Npool.isn(std::get<std::string>(a) + "+" + std::get<std::string>(b)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "+" + std::get<std::string>(b));
        vname = std::get<std::string>(a) + "+" + std::get<std::string>(b);
    }
    else if (Npool.isn(std::get<std::string>(b) + "+" + std::get<std::string>(a)))
    {
        result = Npool.require_variable(std::get<std::string>(b) + "+" + std::get<std::string>(a));
        vname = std::get<std::string>(b) + "+" + std::get<std::string>(a);
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "+" + std::get<std::string>(b));
        vname = std::get<std::string>(a) + "+" + std::get<std::string>(b);
    }

    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        add(std::get<double *>(a), std::get<double *>(b), result, s0["d1"]);
        break;
    case 2:
        add(std::get<double *>(a), std::get<double *>(b), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        add(std::get<double *>(a), std::get<double *>(b), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(vname, s0, result);
}

Variable operator+(Variable a, double b)
{

    double *result;
    std::string vname;
    if (Npool.isn(std::get<std::string>(a) + "+" + std::to_string(b)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "+" + std::to_string(b));
        vname = std::get<std::string>(a) + "+" + std::to_string(b);
    }
    else if (Npool.isn(std::to_string(b) + "+" + std::get<std::string>(a)))
    {
        result = Npool.require_variable(std::to_string(b) + "+" + std::get<std::string>(a));
        vname = std::to_string(b) + "+" + std::get<std::string>(a);        
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "+" + std::to_string(b));
        vname = std::get<std::string>(a) + "+" + std::to_string(b);
    }

    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        add(std::get<double *>(a), b, result, s0["d1"]);
        break;
    case 2:
        add(std::get<double *>(a), b, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        add(std::get<double *>(a), b, result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(vname, s0, result);
}

Variable operator+(double a, Variable b)
{

    double *result;
    std::string vname;
    if (Npool.isn(std::to_string(a) + "+" + std::get<std::string>(b)))
    {
        result = Npool.require_variable(std::to_string(a) + "+" + std::get<std::string>(b));
        vname = std::to_string(a) + "+" + std::get<std::string>(b);
    }
    else if (Npool.isn(std::get<std::string>(b) + "+" + std::to_string(a)))
    {
        result = Npool.require_variable(std::get<std::string>(b) + "+" + std::to_string(a));
        vname = std::get<std::string>(b) + "+" + std::to_string(a);        
    }
    else
    {
        result = Npool.register_variable(std::to_string(a) + "+" + std::get<std::string>(b));
        vname = std::to_string(a) + "+" + std::get<std::string>(b);
    }

    Prop::shape s0 = std::get<Prop::shape>(b);
    switch (Prop::dims(s0))
    {
    case 1:
        add(a, std::get<double *>(b), result, s0["d1"]);
        break;
    case 2:
        add(a, std::get<double *>(b), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        add(a, std::get<double *>(b), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(vname, s0, result);
}

Variable operator*(Variable a, Variable b)
{
    double *result;
    std::string vname;
    if (Npool.isn(std::get<std::string>(a) + "*" + std::get<std::string>(b)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "*" + std::get<std::string>(b));
        vname = std::get<std::string>(a) + "*" + std::get<std::string>(b);
    }
    else if (Npool.isn(std::get<std::string>(b) + "*" + std::get<std::string>(a)))
    {
        result = Npool.require_variable(std::get<std::string>(b) + "*" + std::get<std::string>(a));
        vname = std::get<std::string>(b) + "*" + std::get<std::string>(a);
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "*" + std::get<std::string>(b));
        vname = std::get<std::string>(a) + "*" + std::get<std::string>(b);
    }
    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        mul(std::get<double *>(a), std::get<double *>(b), result, s0["d1"]);
        break;
    case 2:
        mul(std::get<double *>(a), std::get<double *>(b), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        mul(std::get<double *>(a), std::get<double *>(b), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(vname, s0, result);
}

Variable operator*(Variable a, double b)
{

    double *result;
    std::string vname;
    if (Npool.isn(std::get<std::string>(a) + "*" + std::to_string(b)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "*" + std::to_string(b));
        vname = std::get<std::string>(a) + "*" + std::to_string(b);
    }
    else if (Npool.isn(std::to_string(b) + "*" + std::get<std::string>(a)))
    {
        result = Npool.require_variable(std::to_string(b) + "*" + std::get<std::string>(a));
        vname = std::to_string(b) + "*" + std::get<std::string>(a);        
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "*" + std::to_string(b));
        vname = std::get<std::string>(a) + "*" + std::to_string(b);
    }

    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        mul(std::get<double *>(a), b, result, s0["d1"]);
        break;
    case 2:
        mul(std::get<double *>(a), b, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        mul(std::get<double *>(a), b, result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(vname, s0, result);
}

Variable operator*(double a, Variable b)
{

    double *result;
    std::string vname;
    if (Npool.isn(std::to_string(a) + "*" + std::get<std::string>(b)))
    {
        result = Npool.require_variable(std::to_string(a) + "*" + std::get<std::string>(b));
        vname = std::to_string(a) + "*" + std::get<std::string>(b);
    }
    else if (Npool.isn(std::get<std::string>(b) + "*" + std::to_string(a)))
    {
        result = Npool.require_variable(std::get<std::string>(b) + "*" + std::to_string(a));
        vname = std::get<std::string>(b) + "*" + std::to_string(a);        
    }
    else
    {
        result = Npool.register_variable(std::to_string(a) + "*" + std::get<std::string>(b));
        vname = std::to_string(a) + "*" + std::get<std::string>(b);
    }

    Prop::shape s0 = std::get<Prop::shape>(b);
    switch (Prop::dims(s0))
    {
    case 1:
        mul(a, std::get<double *>(b), result, s0["d1"]);
        break;
    case 2:
        mul(a, std::get<double *>(b), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        mul(a, std::get<double *>(b), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(vname, s0, result);
}

Variable operator-(Variable a, Variable b)
{
    double *result;
    if (Npool.isn(std::get<std::string>(a) + "-" + std::get<std::string>(b)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "-" + std::get<std::string>(b));
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "-" + std::get<std::string>(b));
    }
    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        sub(std::get<double *>(a), std::get<double *>(b), result, s0["d1"]);
        break;
    case 2:
        sub(std::get<double *>(a), std::get<double *>(b), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        sub(std::get<double *>(a), std::get<double *>(b), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(std::get<std::string>(a) + "-" + std::get<std::string>(b), s0, result);
}

Variable operator-(Variable a, double b)
{
    double *result;
    if (Npool.isn(std::get<std::string>(a) + "-" + std::to_string(b)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "-" + std::to_string(b));
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "-" + std::to_string(b));
    }
    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        sub(std::get<double *>(a), b, result, s0["d1"]);
        break;
    case 2:
        sub(std::get<double *>(a), b, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        sub(std::get<double *>(a), b, result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(std::get<std::string>(a) + "-" + std::to_string(b), s0, result);
}

Variable operator-(double a, Variable b)
{
    double *result;
    if (Npool.isn(std::to_string(a) + "-" + std::get<std::string>(b)))
    {
        result = Npool.require_variable(std::to_string(a) + "-" + std::get<std::string>(b));
    }
    else
    {
        result = Npool.register_variable(std::to_string(a) + "-" + std::get<std::string>(b));
    }
    Prop::shape s0 = std::get<Prop::shape>(b);
    switch (Prop::dims(s0))
    {
    case 1:
        sub(a, std::get<double *>(b), result, s0["d1"]);
        break;
    case 2:
        sub(a, std::get<double *>(b), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        sub(a, std::get<double *>(b), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(std::to_string(a) + "-" + std::get<std::string>(b), s0, result);
}

Variable operator/(Variable a, Variable b)
{
    double *result;
    if (Npool.isn(std::get<std::string>(a) + "/" + std::get<std::string>(b)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "/" + std::get<std::string>(b));
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "/" + std::get<std::string>(b));
    }
    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        div(std::get<double *>(a), std::get<double *>(b), result, s0["d1"]);
        break;
    case 2:
        div(std::get<double *>(a), std::get<double *>(b), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        div(std::get<double *>(a), std::get<double *>(b), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(std::get<std::string>(a) + "/" + std::get<std::string>(b), s0, result);
}

Variable operator/(Variable a, double b)
{
    double *result;
    if (Npool.isn(std::get<std::string>(a) + "/" + std::to_string(b)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "/" + std::to_string(b));
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "/" + std::to_string(b));
    }
    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        div(std::get<double *>(a), b, result, s0["d1"]);
        break;
    case 2:
        div(std::get<double *>(a), b, result, s0["d1"], s0["d2"]);
        break;
    case 3:
        div(std::get<double *>(a), b, result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(std::get<std::string>(a) + "/" + std::to_string(b), s0, result);
}

Variable operator/(double a, Variable b)
{
    double *result;
    if (Npool.isn(std::to_string(a) + "/" + std::get<std::string>(b)))
    {
        result = Npool.require_variable(std::to_string(a) + "/" + std::get<std::string>(b));
    }
    else
    {
        result = Npool.register_variable(std::to_string(a) + "/" + std::get<std::string>(b));
    }
    Prop::shape s0 = std::get<Prop::shape>(b);
    switch (Prop::dims(s0))
    {
    case 1:
        div(a, std::get<double *>(b), result, s0["d1"]);
        break;
    case 2:
        div(a, std::get<double *>(b), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        div(a, std::get<double *>(b), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple(std::to_string(a) + "/" + std::get<std::string>(b), s0, result);
}



Variable sin(Variable a)
{
    double *result;
    if (Npool.isn("sin" + std::get<std::string>(a)))
    {
        result = Npool.require_variable("sin" + std::get<std::string>(a));
    }
    else
    {
        result = Npool.register_variable("sin" + std::get<std::string>(a));
    }
    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        sin(std::get<double *>(a), result, s0["d1"]);
        break;
    case 2:
        sin(std::get<double *>(a), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        sin(std::get<double *>(a), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple("sin" + std::get<std::string>(a), s0, result);
}

Variable cos(Variable a)
{
    double *result;
    if (Npool.isn("cos" + std::get<std::string>(a)))
    {
        result = Npool.require_variable("cos" + std::get<std::string>(a));
    }
    else
    {
        result = Npool.register_variable("cos" + std::get<std::string>(a));
    }
    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        cos(std::get<double *>(a), result, s0["d1"]);
        break;
    case 2:
        cos(std::get<double *>(a), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        cos(std::get<double *>(a), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple("cos" + std::get<std::string>(a), s0, result);
}

Variable tan(Variable a)
{
    double *result;
    if (Npool.isn("tan" + std::get<std::string>(a)))
    {
        result = Npool.require_variable("tan" + std::get<std::string>(a));
    }
    else
    {
        result = Npool.register_variable("tan" + std::get<std::string>(a));
    }
    Prop::shape s0 = std::get<Prop::shape>(a);
    switch (Prop::dims(s0))
    {
    case 1:
        tan(std::get<double *>(a), result, s0["d1"]);
        break;
    case 2:
        tan(std::get<double *>(a), result, s0["d1"], s0["d2"]);
        break;
    case 3:
        tan(std::get<double *>(a), result, s0["d1"], s0["d2"], s0["d3"]);
        break;   
    default:
        break;
    }
    return std::make_tuple("tan" + std::get<std::string>(a), s0, result);
}
/*
Variable laplace(Variable a)
{
    double *result;
    if (Npool.isn("laplace" + std::get<std::string>(a)))
    {
        result = Npool.require_variable("laplace" + std::get<std::string>(a));
    }
    else
    {
        result = Npool.register_variable("laplace" + std::get<std::string>(a));
    }
    // The boundary is not processed
    Prop::shape s0 = std::get<Prop::shape>(a);
    laplace(std::get<double*>(a), result, s0["d1"], s0["d2"]);
    return std::make_tuple("laplace" + std::get<std::string>(a), s0, result);  
}
*/

Variable parti_diff(Variable a, int i, int j)
{
    double *result;
    if (Npool.isn(std::get<std::string>(a) + "(i)" + std::to_string(i) + "(j)" + std::to_string(j)))
    {
        result = Npool.require_variable(std::get<std::string>(a) + "(i)" + std::to_string(i) + "(j)" + std::to_string(j));
    }
    else
    {
        result = Npool.register_variable(std::get<std::string>(a) + "(i)" + std::to_string(i) + "(j)" + std::to_string(j));
    }       
    Prop::shape s0 = std::get<Prop::shape>(a);
    parti_diff(std::get<double*>(a), result, s0["d1"], s0["d2"], i, j);
    return std::make_tuple(std::get<std::string>(a) + "(i)" + std::to_string(i) + "(j)" + std::to_string(j), s0, result);
}

Variable zero_boundary(Variable a)
{
    Prop::shape s0 = std::get<Prop::shape>(a);
    zero_boundary(std::get<double*>(a), s0["d1"], s0["d2"]);
    return a;
}

Variable laplace(Variable a)
{
    return parti_diff(a, 1, 0) + parti_diff(a, -1, 0) + parti_diff(a, 0, 1) + parti_diff(a, 0, -1) - 4 * a;
}
