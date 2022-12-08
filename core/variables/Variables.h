#ifndef VARIABLES_H
#define VARIABLES_H
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

// zero_boundary马上被移除
Variable zero_boundary(const Variable &a);

Variable laplace(const Variable &a);

int operator<<(Variable &a, const Variable &b);


Variable read_from_netcdf(char* filepath, char* name);


Variable init_Variable(std::string name, Prop::shape s, const void *v);

// 二维矩阵的乘法
Variable mul2(const Variable& a, const Variable& b);
Variable ones2(Prop::shape s);

void print_info(const Variable& v);

#endif
