#include <core/variables/Variables.h>

namespace boundary
{
    Variable zero(Variable &a);
    /**
     * @brief 常数边界
     * 常数边界的值来自于初始化条件，因此常数边界要求提供初始化输入
     * @param a
     * @return Variable
     */
    Variable cst(Variable &a, const Variable &init);
};