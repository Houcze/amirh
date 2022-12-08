#include <core/boundary/boundary.h>
#include <core/function/Prop.h>
#include <core/function/func.h>

Variable boundary::zero(Variable &a)
{
    Prop::shape s0 = a.shape;
    zero_boundary_dev(a.val, s0["d1"], s0["d2"]);
    return a;
}

Variable cst(Variable &a, const Variable &init)
{
    return a;
}
