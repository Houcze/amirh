#ifndef PROP_H
#define PROP_H
#include <map>
#include <string>

namespace Prop
{
    using shape = std::map<std::string, int>;
    using scalar = std::map<std::string, int>;
    int size(shape);
    int dims(shape);
    void print(Prop::shape);
    std::string to_string(Prop::shape);
}

#endif