#include <core/domain/domain.h>
#include <string>
#include <core/variables/Variables.h>

domain::arakawac::arakawac(
    const double w_b, 
    const double e_b, 
    const double n_b, 
    const double s_b, 
    const double u_b, 
    const double l_b, 
    double *level1, 
    double *level2, 
    double *level3, 
    Prop::shape level_shape
)
{
    wb = w_b;
    eb = e_b;
    nb = n_b;
    sb = s_b;
    ub = u_b;
    lb = l_b;
    lev1 = level1;
    lev2 = level2;
    lev3 = level3;
    levpe = level_shape;
}

Variable domain::arakawac::alloc(std::string name, int types)
{
    
}