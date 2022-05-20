#include <cstddef>

namespace netcdf {
    int get_size(size_t *size, size_t *d, char *filepath, char* varname);
    int read(double *result, size_t *shape, char* filepath, char* varname);
}


namespace data = netcdf;