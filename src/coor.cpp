#include <tensor.h>
#define pi 3.1415926535

/*
*  spherical coordinate system
*/

namespace domain {
    
    class coor
    {
    public:
        double r;
        double theta;
        double phi;
        /*
        * https://zh.m.wikipedia.org/zh-hans/File:3D_Spherical.svg
        * r >= 0; 0<=theta<=pi; 0<=phi<=2pi
        */
        coor(double, double, double);
        ~coor();
        coor next_r(double);
        coor next_theta(double);
        coor next_phi(double);
    };

        
}

domain::coor::coor(double r_input, double theta_input, double phi_input)
{
    r = r_input;
    theta = theta_input;
    phi = phi_input;
    theta = theta % pi;
    phi = phi % (2 * pi);
}
    
domain::coor::~coor()
{
    ;
}

domain::coor 
domain::coor::next_r(double step)
{
    return domain::coor(r + step, theta, phi);
}

domain::coor 
domain::coor::next_theta(double step)
{
    if((theta + step)> pi)
    {
        theta = theta - pi;
    }
    return domain::coor(r, theta, phi);
}

domain::coor 
domain::coor::next_phi(double step)
{
    if((phi + step)> 2 * pi)
    {
        phi = theta - 2 * pi;
    }
    return domain::coor(r, theta, phi);
}