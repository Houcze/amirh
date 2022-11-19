int add(double* x, double* y, double* result, int N1);
int add(double* x, double* y, double* result, int N1, int N2);
int add(double* x, double* y, double* result, int N1, int N2, int N3);

int sub(double* x, double* y, double* result, int N1);
int sub(double* x, double* y, double* result, int N1, int N2);
int sub(double* x, double* y, double* result, int N1, int N2, int N3);

int mul(double* x, double* y, double* result, int N1);
int mul(double* x, double* y, double* result, int N1, int N2);
int mul(double* x, double* y, double* result, int N1, int N2, int N3);

int div(double* x, double* y, double* result, int N1);
int div(double* x, double* y, double* result, int N1, int N2);
int div(double* x, double* y, double* result, int N1, int N2, int N3);

int add(double* x, double y, double* result, int N1);
int add(double* x, double y, double* result, int N1, int N2);
int add(double* x, double y, double* result, int N1, int N2, int N3);

int sub(double* x, double y, double* result, int N1);
int sub(double* x, double y, double* result, int N1, int N2);
int sub(double* x, double y, double* result, int N1, int N2, int N3);

int mul(double* x, double y, double* result, int N1);
int mul(double* x, double y, double* result, int N1, int N2);
int mul(double* x, double y, double* result, int N1, int N2, int N3);

int div(double* x, double y, double* result, int N1);
int div(double* x, double y, double* result, int N1, int N2);
int div(double* x, double y, double* result, int N1, int N2, int N3);

int laplace(double* phi, double* result, int N1, int N2);

int expectation(double* x, double* y, double* result, int w1, int w2, int N1, int N2);
int expectation(double* x, double* y, double* z, double* result, int w1, int w2, int w3, int N1, int N2, int N3);

int sin(double* input, double* result, int N1, int N2, int N3);
int cos(double* input, double* result, int N1, int N2, int N3);
int tan(double* input, double* result, int N1, int N2, int N3);

int sin(double* input, double* result, int N1, int N2);
int cos(double* input, double* result, int N1, int N2);
int tan(double* input, double* result, int N1, int N2);

int sin(double* input, double* result, int N);
int cos(double* input, double* result, int N);
int tan(double* input, double* result, int N);
