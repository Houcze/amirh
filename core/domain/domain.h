#include <core/variables/Variables.h>
#include <core/function/Prop.h>
#include <string>
#include <map>

namespace domain
{
    class arakawac
    {
    private:
        double wb;
        double eb;
        double nb;
        double sb;
        double ub;
        double lb;
        /**
         * @brief lev1, lev2, lev3是和风场坐标相关的信息，为不规则的离散化留下可能
         */
        double *lev1;
        double *lev2;
        double *lev3;
        Prop::shape levpe;

    public:
        arakawac(
            const double w_b,
            const double e_b,
            const double n_b,
            const double s_b,
            const double u_b,
            const double l_b,
            double *level1,
            double *level2,
            double *level3,
            Prop::shape level_shape
        );

        Variable alloc(std::string name, int types);
    };
};