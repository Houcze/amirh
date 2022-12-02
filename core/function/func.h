#ifndef FUNC_H
#define FUNC_H
#include <vector>
#include <string>

int add(double *x, double *y, double *result, int N1);
int add(double *x, double *y, double *result, int N1, int N2);
int add(double *x, double *y, double *result, int N1, int N2, int N3);

int add(double *x, double y, double *result, int N1);
int add(double *x, double y, double *result, int N1, int N2);
int add(double *x, double y, double *result, int N1, int N2, int N3);

int add(double x, double *y, double *result, int N1);
int add(double x, double *y, double *result, int N1, int N2);
int add(double x, double *y, double *result, int N1, int N2, int N3);

int sub(double *x, double *y, double *result, int N1);
int sub(double *x, double *y, double *result, int N1, int N2);
int sub(double *x, double *y, double *result, int N1, int N2, int N3);

int sub(double *x, double y, double *result, int N1);
int sub(double *x, double y, double *result, int N1, int N2);
int sub(double *x, double y, double *result, int N1, int N2, int N3);

int sub(double x, double *y, double *result, int N1);
int sub(double x, double *y, double *result, int N1, int N2);
int sub(double x, double *y, double *result, int N1, int N2, int N3);

int mul(double *x, double *y, double *result, int N1);
int mul(double *x, double *y, double *result, int N1, int N2);
int mul(double *x, double *y, double *result, int N1, int N2, int N3);

int mul(double *x, double y, double *result, int N1);
int mul(double *x, double y, double *result, int N1, int N2);
int mul(double *x, double y, double *result, int N1, int N2, int N3);

int mul(double x, double *y, double *result, int N1);
int mul(double x, double *y, double *result, int N1, int N2);
int mul(double x, double *y, double *result, int N1, int N2, int N3);

int div(double *x, double *y, double *result, int N1);
int div(double *x, double *y, double *result, int N1, int N2);
int div(double *x, double *y, double *result, int N1, int N2, int N3);

int div(double *x, double y, double *result, int N1);
int div(double *x, double y, double *result, int N1, int N2);
int div(double *x, double y, double *result, int N1, int N2, int N3);

int div(double x, double *y, double *result, int N1);
int div(double x, double *y, double *result, int N1, int N2);
int div(double x, double *y, double *result, int N1, int N2, int N3);


int laplace(double *phi, double *result, int N1, int N2);

int expectation(double *x, double *y, double *result, int w1, int w2, int N1, int N2);
int expectation(double *x, double *y, double *z, double *result, int w1, int w2, int w3, int N1, int N2, int N3);

int sin(double *input, double *result, int N1, int N2, int N3);
int cos(double *input, double *result, int N1, int N2, int N3);
int tan(double *input, double *result, int N1, int N2, int N3);

int sin(double *input, double *result, int N1, int N2);
int cos(double *input, double *result, int N1, int N2);
int tan(double *input, double *result, int N1, int N2);

int sin(double *input, double *result, int N);
int cos(double *input, double *result, int N);
int tan(double *input, double *result, int N);

int get_gpuc();
int ones(double *arr, int N);
int ones(double *arr, int N1, int N2);
int ones(double *arr, int N1, int N2, int N3);

int zeros(double *arr, int N);
int zeros(double *arr, int N1, int N2);
int zeros(double *arr, int N1, int N2, int N3);

int set_cons(double *arr, double value, int N);
int set_cons(double *arr, double value, int N1, int N2);
int set_cons(double *arr, double value, int N1, int N2, int N3);

int prod(std::vector<int> p);
int seq_add(std::vector<double *> vlist, double *result, int N1, int N2, int N3);
int seq_mul(std::vector<double *> vlist, double *result, int N1, int N2, int N3);
int seq_add(std::vector<double *> vlist, double *result, int N1, int N2);
int seq_mul(std::vector<double *> vlist, double *result, int N1, int N2);

int save_txt(double *var, std::string filepath, int N1, int N2);
int save_txt_dev(double *var, std::string filepath, int N1, int N2);

int parti_diff(double *phi, double *result, int N1, int N2, int i, int j);
int zero_boundary(double* result, int N1, int N2);

#endif
