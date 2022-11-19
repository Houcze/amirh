#include <core/function/Prop.h>
#include <string>


namespace std 
{
    using Name = std::string;
    class Variable
    {
    private:
        Prop::shape s_;
        Name vname;      
    public:
        double* ds;
        Variable::Variable(Prop::shape s, std::Name n);
        Variable::Variable(Prop::shape s);
        Variable::~Variable();
        int size();
        int dims();
        int register_name(std::Name);
    };
}