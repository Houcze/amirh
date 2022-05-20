#include <iostream>
#include <cstdlib>
#include <cstring>
#include <netcdf.h>


namespace netcdf {
    int get_size(size_t *size, size_t *d, char *filepath, char* varname)
    {
    	int status;
        int ncid;
        int varid;
        nc_type vartype;
        int varndims;
        int vardimids[NC_MAX_VAR_DIMS];
        int varnatts;
        size_t recs;
        char recname[NC_MAX_NAME+1];

        status = nc_open(filepath, NC_NOWRITE, &ncid);
        status = nc_inq_varid(ncid, varname, &varid);
        status = nc_inq_var(ncid, varid, 0, &vartype, &varndims, vardimids, &varnatts);

        size_t *dims;
        dims = (size_t *) std::malloc(varndims * sizeof(size_t));
        std::string *dnames;
        dnames = (std::string *) std::malloc(varndims * sizeof(std::string));


        for(int serial_num=0; serial_num < varndims; serial_num++)
        {
            status = nc_inq_dim(ncid, vardimids[serial_num], recname, &recs);
            // std::cout << recname << ' ' << recs << '\n';
            dnames[serial_num].assign(recname);
            dims[serial_num] = recs;
    		*size *= recs;
    		// std::cout << "recs is " << recs << ", " << "size is " << *size << std::endl;
        }
    	nc_close(ncid);
    	*d = varndims;
    	return EXIT_SUCCESS;
    }



    int read(double *result, size_t *shape, char* filepath, char* varname)
    {
        int status;
        int ncid;
        int varid;
        nc_type vartype;
        int varndims;
        int vardimids[NC_MAX_VAR_DIMS];
        int varnatts;
        size_t recs;
        char recname[NC_MAX_NAME+1];
        
        int size{1};
        status = nc_open(filepath, NC_NOWRITE, &ncid);
        status = nc_inq_varid(ncid, varname, &varid);
        status = nc_inq_var(ncid, varid, 0, &vartype, &varndims, vardimids, &varnatts);

        size_t *dims;
        dims = (size_t *) std::malloc(varndims * sizeof(size_t));
        std::string *dnames;
        dnames = (std::string *) std::malloc(varndims * sizeof(std::string));

        for(int serial_num=0; serial_num < varndims; serial_num++)
        {
            status = nc_inq_dim(ncid, vardimids[serial_num], recname, &recs);
            dnames[serial_num].assign(recname);
            dims[serial_num] = recs;
            size *= recs;
        }
        nc_get_var_double(ncid, varid, result);
        std::cout << "Success Load " << varname << '\n';
        status = nc_close(ncid);
        memcpy(shape, dims, sizeof(dims));
        return EXIT_SUCCESS;
    }
}
