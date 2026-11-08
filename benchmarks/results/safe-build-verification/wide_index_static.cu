#include "helper.h"
#include <iostream>
static_assert(IDX2C(580,5631999,581)==std::size_t{3272191999ULL}, "Matrix offsets must not overflow signed32-bit");
int main() { std::cout << IDX2C(580,5631999,581) << '\n'; }
