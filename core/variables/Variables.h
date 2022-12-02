#include <core/function/Prop.h>

#include <string>
#include <tuple>

using Variable = std::tuple<std::string, Prop::shape, double *>;

Variable operator+(Variable a, Variable b);
Variable operator-(Variable a, Variable b);
Variable operator*(Variable a, Variable b);
Variable operator/(Variable a, Variable b);

Variable operator+(double a, Variable b);
Variable operator-(double a, Variable b);
Variable operator*(double a, Variable b);
Variable operator/(double a, Variable b);

Variable operator+(Variable a, double b);
Variable operator-(Variable a, double b);
Variable operator*(Variable a, double b);
Variable operator/(Variable a, double b);

Variable sin(Variable a);
Variable cos(Variable a);
Variable tan(Variable a);

Variable parti_diff(Variable a, int i, int j);
Variable laplace(Variable a);
Variable zero_boundary(Variable a);
