#ifndef PROP_H
#define PROP_H
#include <map>
#include <string>

namespace Prop{
    using shape = std::map<std::string, int>;
    int size(shape);
    int dims(shape);
}

int Prop::size(Prop::shape p)
{
    int size{1};
    for(const auto& it : p)
    {
        size *= (it.second);
    }
    return size;
}

int Prop::dims(Prop::shape p)
{
    return p.size();
}

/*
bool operator==(Prop::shape s0, Prop::shape s1)
{
    if(Prop::size(s0) == Prop::size(s1))
    {
        for(int dn=0; dn <= Prop::dims(s0); dn++)
        {
            if(s0["d" + std::to_string(dn)] != s1["d" + std::to_string(dn)])
            {
                return false;
            }
        }
        return true;
    }
    return false;
}
*/

#endif