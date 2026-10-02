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

# End-to-end checks against the reference pickles in validate/: a fast s = 7
# prefix test under ctest (skipped without references), and a full s = 9
# byte-for-byte comparison as the `validate` target.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  set(NLCE_REF_DIR "${CMAKE_SOURCE_DIR}/validate" CACHE PATH "Reference series pickles")
  add_test(NAME regression
    COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/regression.py
      --binary $<TARGET_FILE:nlce_run> --nsites 7 --ref ${NLCE_REF_DIR})
  set_tests_properties(regression PROPERTIES SKIP_RETURN_CODE 77)

  set(NLCE_VALIDATE_DIR "${CMAKE_BINARY_DIR}/validate_s9")
  add_custom_target(validate
    COMMAND $<TARGET_FILE:nlce_run> run --nsites 9 --out ${NLCE_VALIDATE_DIR}
    COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/to_pickles.py ${NLCE_VALIDATE_DIR} ${NLCE_VALIDATE_DIR}/pkl
    COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/compare_pickles.py --bytes ${NLCE_VALIDATE_DIR}/pkl ${NLCE_REF_DIR}
    DEPENDS nlce_run
    USES_TERMINAL
    COMMENT "Full s = 9 run compared byte for byte against ${NLCE_REF_DIR}")
endif()
