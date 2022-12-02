#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
 
// 要观览的 variant
using var_t = std::variant<int, long, double, std::string>;
 
// 观览器 #3 的辅助常量
template<class> inline constexpr bool always_false_v = false;
 
// 观览器 #4 的辅助类型
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// 显式推导指引（ C++20 起不需要）
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
 
int main() {
    std::vector<var_t> vec = {10, 15l, 1.5, "hello"};
    
    for (auto&& v: vec) {
        // 4. 另一种类型匹配观览器：有三个重载的 operator() 的类
        // 注：此情况下 '(auto arg)' 模板 operator() 将绑定到 'int' 与 'long' ，
        //    但若它不存在则 '(double arg)' operator() *亦将* 绑定到 'int' 与 'long' ，
        //    因为两者均可隐式转换成 double 。使用此形式时应留心以正确处理隐式转换。
        std::visit(overloaded {
            [](auto arg) { std::cout << arg << ' '; },
            [](double arg) { std::cout << std::fixed << arg << ' '; },
            [](const std::string& arg) { std::cout << std::quoted(arg) << ' '; },
        }, v);
    }
}