#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace pynanovdb {

void defineMathModule(nb::module_& m);

} // namespace pynanovdb