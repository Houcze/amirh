#include <core/solver/solver.h>

Variable solver::phi1(const Variable &phi, double h, Variable (*R)(const Variable &))
{
    Variable result = R(phi);
    result = result * (h / 3);
    result = result + phi;
    return result;
}

Variable solver::phi2(const Variable &phi, double h, Variable (*R)(const Variable &))
{
    Variable result = solver::phi1(phi, h, R);
    result = R(result);
    result = result * (h / 2);
    result = result + phi;
    return result;
}

Variable solver::Rk3(Variable phi, double h, Variable (*R)(const Variable &))
{
    Variable result = solver::phi2(phi, h, R);
    result = R(result);
    result = result * h;
    result = result + phi;
    return result;
}
