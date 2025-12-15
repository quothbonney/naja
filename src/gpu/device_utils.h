#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

namespace naja::gpu {
    void set_device(int device);
    std::vector<std::string> list_devices();
}
