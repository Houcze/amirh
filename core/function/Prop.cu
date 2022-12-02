#include <core/function/Prop.h>

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
