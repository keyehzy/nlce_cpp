cmake_minimum_required(VERSION 3.24)
project(nlce_cpp LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NLCE_NATIVE "Tune for the build machine's CPU" ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLINT REQUIRED IMPORTED_TARGET flint)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
find_package(Boost 1.81 REQUIRED CONFIG)
find_package(Threads REQUIRED)

add_library(nlce STATIC
  src/modp.cpp
  src/lattice.cpp
  src/graph.cpp
  src/geometry.cpp
  src/sector.cpp
  src/cluster.cpp
  src/pipeline.cpp
  src/reconstruct.cpp
)
target_include_directories(nlce PUBLIC src)
target_link_libraries(nlce PUBLIC PkgConfig::FLINT PkgConfig::GMP Boost::headers Threads::Threads)
target_compile_options(nlce PUBLIC -Wall -Wextra -Wno-sign-compare)

if(NLCE_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=native" NLCE_HAS_MCPU_NATIVE)
  if(NLCE_HAS_MCPU_NATIVE)
    target_compile_options(nlce PUBLIC -mcpu=native)
  endif()
endif()

add_executable(nlce_run src/main.cpp)
target_link_libraries(nlce_run PRIVATE nlce)

enable_testing()
add_executable(nlce_tests tests/unit_tests.cpp)
target_link_libraries(nlce_tests PRIVATE nlce)
add_test(NAME unit COMMAND nlce_tests)
