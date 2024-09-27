cmake_minimum_required(VERSION 3.10)
project(duckdb_python_udf)
set(CMAKE_CXX_STANDARD 14)

find_package(pybind11 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/../../src/include ${pybind11_INCLUDE_DIRS})

# 添加可执行文件
add_executable(example main.cpp)

# 链接 DuckDB 库
find_library(DUCKDB_LIB duckdb PATHS ${CMAKE_SOURCE_DIR}/../../build/src)
target_link_libraries(example ${DUCKDB_LIB} pybind11::embed)







