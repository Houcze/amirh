#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <io/netcdf>
#include <core/mem.h>
#include <core/function/func.h>
#include <core/solver/solver.h>
#include <core/variables/Variables.h>
#include <core/boundary/boundary.h>

/*
 * This R function is user defined
 */
Variable R(const Variable &phi)
{
    Variable result = parti_diff(phi, 1, 0) + parti_diff(phi, -1, 0) + parti_diff(phi, 0, 1) + parti_diff(phi, 0, -1) - 4 * phi;
    boundary::zero(result);
    return result;
}

int main(void)
{
    /**
     * t, N1, N2, N3 will be got from the file
     */
    int t{2};
    int N1{41};
    int N2{360};
    double h{30};
    Prop::shape s0 = {{"d1", N1}, {"d2", N2}};

    // 显存池也可以有显存池
    // Npool.register_shape(s0);
    // ------------------------------------------------------------------------------------------------

    // Use python to process this, generate the right name.
    char varname[] = "HGT_500mb";
    std::vector<Variable> hgt500list;    
    for (auto &p : std::filesystem::directory_iterator("./ds"))
    {
        hgt500list.push_back(read_from_netcdf(p.path().string().data(), varname));
    }
    // ------------------------------------------------------------------------------------------------
    // 这一部分交给预处理，用python完成
    // Variable lat = read_from_netcdf("./lat.nc", "lat");
    // Variable lon = read_from_netcdf("./lon.nc", "lon");
    /**
     * 生成经纬度网格（带编号的）相关代码
     */
    /* 12-5 00::31 这个地方应该提供一种不从netcdf文件取得的方式，可以自动生成, 这需要一个linspace函数
     */
    char filepath_1[] = "./ds/amirh_20200531_06_00_00.nc";
    char filepath_2[] = "./ds/amirh_20200531_12_00_00.nc";
    
    Variable lon = read_from_netcdf(filepath_1, "longitude");
    Variable lat = read_from_netcdf(filepath_1, "latitude");
    
    //Variable lon_m = mul2(ones2(Prop::shape{{"d1", lat.shape["d1"]}, {"d2", 1}}), lon);
    //Variable lat_m = mul2(lat, ones2(Prop::shape{{"d1", 1}, {"d2", lon.shape["d2"]}})); 

    lon = mul2(ones2(Prop::shape{{"d1", lat.shape["d1"]}, {"d2", 1}}), lon);
    lat = mul2(lat, ones2(Prop::shape{{"d1", 1}, {"d2", lon.shape["d2"]}})); 

    // std::cout << "The size is " << Prop::dims(lon.shape) << std::endl;
    //----------------------------------------------------------------------------------------------//

    Variable pzpt = hgt500list[1] - hgt500list[0];
    pzpt = pzpt / (60 * 60 * 6);
    /* 注意上面这个h还需要修改 */

    for (int i = 0; i < 5; i++)
    {
        std::cout << "Round " << i + 1 << std::endl;
        pzpt = solver::Rk3(pzpt, h, R);   
        if (i >= 0)
        {
            save_txt_dev(pzpt.val, "./result/" + std::to_string(i + 1) + ".txt", N1, N2);
        }
    }
}
