cmake_minimum_required(VERSION 3.14)
project(ConvertSklearnToONNX)

set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})

add_executable(convert convert.cpp)

target_link_libraries(convert PRIVATE pybind11::embed ${Python3_LIBRARIES})
