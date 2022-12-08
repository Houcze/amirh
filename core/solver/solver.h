#include <core/variables/Variables.h>

namespace solver{
    Variable phi1(const Variable &phi, double h, Variable (*R)(const Variable &));
    Variable phi2(const Variable &phi, double h, Variable (*R)(const Variable &));
    Variable Rk3(Variable phi, double h, Variable (*R)(const Variable &));
};
