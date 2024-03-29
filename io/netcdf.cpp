#include <io/netcdf>

int netcdf::get_size(size_t *size, size_t *d, char *filepath, char* varname)
{
    /*
        计划删除此函数，尽快移除所有接口
    */
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

    // size_t *dims;
    // dims = (size_t *) malloc(varndims * sizeof(size_t));
    // std::string *dnames;
    // dnames = (std::string *) std::malloc(varndims * sizeof(std::string));
    // std::string dnames[varndims];
    
    std::map<char*, int> prop;

    for(int serial_num=0; serial_num < varndims; serial_num++)
    {
        status = nc_inq_dim(ncid, vardimids[serial_num], recname, &recs);
        // std::cout << recname << ' ' << recs << '\n';
        // dnames[serial_num].assign(recname);
        prop[recname] = recs;
        std::cout << recname << " : " << recs << '\n';
        // dims[serial_num] = recs;
        *size *= recs;
        // std::cout << "dnames " << serial_num << " is " << dnames[serial_num] << '\n';
        // std::cout << "recs is " << recs << ", " << "size is " << *size << std::endl;
    }
    nc_close(ncid);
    // *d = varndims;
    *d = prop.size();
    return status;
}


int netcdf::ds_prop(Prop::shape* p, char *filepath, char* varname)
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


    for(int serial_num=0; serial_num < varndims; serial_num++)
    {
        status = nc_inq_dim(ncid, vardimids[serial_num], recname, &recs);
        (*p)[recname] = recs;
    }

    nc_close(ncid);
    return status;
}


int netcdf::ds(double *result, char* filepath, char* varname)
{
    int status;
    int ncid;
    int varid;
    nc_type vartype;
    int varndims;
    int vardimids[NC_MAX_VAR_DIMS];
    int varnatts;

        
    status = nc_open(filepath, NC_NOWRITE, &ncid);
    status = nc_inq_varid(ncid, varname, &varid);
    status = nc_inq_var(ncid, varid, 0, &vartype, &varndims, vardimids, &varnatts);


    nc_get_var_double(ncid, varid, result);
    std::cout << "Success Load " << varname << '\n';
    status = nc_close(ncid);
    return status;
}
