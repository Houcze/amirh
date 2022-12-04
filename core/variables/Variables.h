#include <core/function/Prop.h>
#include <string>

// using Variable = std::tuple<std::string, Prop::shape, double *>;

struct Variable
{
    std::string name;
    Prop::shape shape;
    double *val;
    Variable(std::string, Prop::shape, double *);
    Variable(const Variable &);
    int operator=(Variable);
};

Variable make_variable(std::string name, Prop::shape s, double *val);

Variable operator+(const Variable &a, const Variable &b);
Variable operator-(const Variable &a, const Variable &b);
Variable operator*(const Variable &a, const Variable &b);
Variable operator/(const Variable &a, const Variable &b);

Variable operator+(double a, const Variable &b);
Variable operator-(double a, const Variable &b);
Variable operator*(double a, const Variable &b);
Variable operator/(double a, const Variable &b);

Variable operator+(const Variable &a, double b);
Variable operator-(const Variable &a, double b);
Variable operator*(const Variable &a, double b);
Variable operator/(const Variable &a, double b);

Variable sin(const Variable &a);
Variable cos(const Variable &a);
Variable tan(const Variable &a);

Variable parti_diff(const Variable &a, int i, int j);
Variable zero_boundary(const Variable &a);

Variable laplace(const Variable &a);

int operator<<(Variable &a, const Variable &b);


Variable read_from_netcdf(char* filepath, char* name);
