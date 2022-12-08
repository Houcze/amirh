#include <core/function/Prop.h>
#include <iostream>

int Prop::size(Prop::shape p)
{
    int size{1};
    for (const auto &it : p)
    {
        size *= (it.second);
    }
    return size;
}

int Prop::dims(Prop::shape p)
{
    return p.size();
}

void Prop::print(Prop::shape p)
{
    for (const auto &it : p)
    {
        std::cout << it.first << ": " << it.second << std::endl;
    }  
}

std::string Prop::to_string(Prop::shape p)
{
    std::string name = "";
    for (const auto &it : p)
    {
        name += it.first;
        name += std::to_string(it.second);
    }
    return name;    
}