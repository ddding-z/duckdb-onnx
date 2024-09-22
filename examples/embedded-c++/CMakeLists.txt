# cmake_minimum_required(VERSION 2.8.12)
# project(example-c++)

# set(CMAKE_CXX_STANDARD 14)

# # 包含 DuckDB 和 ONNX Runtime 的头文件
# include_directories(${CMAKE_SOURCE_DIR}/../../src/include)
# include_directories(${CMAKE_SOURCE_DIR}/../../third_party/onnxruntime/include)
# include_directories(${CMAKE_SOURCE_DIR}/../../third_party/onnx_optimizer/onnx_optimizer) 

# # 包含 Protobuf 头文件
# find_package(Protobuf REQUIRED)
# include_directories(${Protobuf_INCLUDE_DIRS})

# # 定义 ONNX 的宏
# add_definitions(-DONNX_ML=1)
# add_definitions(-DONNX_NAMESPACE=onnx)

# # 添加可执行文件
# add_executable(example main.cpp)

# # 链接 DuckDB 库
# find_library(DUCKDB_LIB duckdb PATHS ${CMAKE_SOURCE_DIR}/../../build/src)
# if(DUCKDB_LIB)
#     message(STATUS "DuckDB library found: ${DUCKDB_LIB}")
# else()
#     message(FATAL_ERROR "DuckDB library not found.")
# endif()
# target_link_libraries(example ${DUCKDB_LIB})

# # 链接 ONNX Runtime 库
# find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${CMAKE_SOURCE_DIR}/../../third_party/onnxruntime/lib)
# if(ONNXRUNTIME_LIB)
#     message(STATUS "ONNX Runtime library found: ${ONNXRUNTIME_LIB}")
# else()
#     message(FATAL_ERROR "ONNX Runtime library not found.")
# endif()
# target_link_libraries(example ${ONNXRUNTIME_LIB})

# # 链接 ONNX 和 ONNX Proto 库
# find_library(ONNX_LIB onnx PATHS ${CMAKE_SOURCE_DIR}/../../third_party/onnxruntime/lib)
# find_library(ONNX_PROTO_LIB onnx_proto PATHS ${CMAKE_SOURCE_DIR}/../../third_party/onnxruntime/lib)
# target_link_libraries(example ${ONNX_LIB} ${ONNX_PROTO_LIB})

# # 链接 Protobuf 库
# target_link_libraries(example ${Protobuf_LIBRARIES})


cmake_minimum_required(VERSION 3.10)
project(example-c++)

set(CMAKE_CXX_STANDARD 14)

# 包含 DuckDB 和 ONNX Runtime 的头文件
include_directories(${CMAKE_SOURCE_DIR}/../../src/include)
# include_directories(${CMAKE_SOURCE_DIR}/../../third_party/onnxruntime/include)

# 添加可执行文件
add_executable(example main.cpp)

# # 包含 Protobuf 头文件
# find_package(Protobuf REQUIRED)
# include_directories(${Protobuf_INCLUDE_DIRS})

# # 链接 Protobuf 库
# target_link_libraries(example ${Protobuf_LIBRARIES})

# # # 定义 ONNX 的宏
# target_compile_definitions(example PRIVATE ONNX_ML=1)
# target_compile_definitions(example PRIVATE DONNX_NAMESPACE=onnx)

# # 查找并链接 ONNX 库
# find_package(ONNX REQUIRED)
# include_directories(${ONNX_INCLUDE_DIRS})
# target_link_libraries(example ${ONNX_LIBRARIES})


# # 查找并链接 ONNX Optimizer 库
# find_package(ONNXOptimizer REQUIRED)
# include_directories(${ONNXOptimizer_INCLUDE_DIRS})
# target_link_libraries(example ${ONNXOptimizer_LIBRARIES})


# target_link_libraries(example ${ONNXOptimizer_LIBRARIES})

# 链接 DuckDB 库
find_library(DUCKDB_LIB duckdb PATHS ${CMAKE_SOURCE_DIR}/../../build/src)
if(DUCKDB_LIB)
    message(STATUS "DuckDB library found: ${DUCKDB_LIB}")
else()
    message(FATAL_ERROR "DuckDB library not found.")
endif()
target_link_libraries(example ${DUCKDB_LIB})


# # 链接 ONNX Runtime 库
# find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${CMAKE_SOURCE_DIR}/../../third_party/onnxruntime/lib)
# if(ONNXRUNTIME_LIB)
#     message(STATUS "ONNX Runtime library found: ${ONNXRUNTIME_LIB}")
# else()
#     message(FATAL_ERROR "ONNX Runtime library not found.")
# endif()
# target_link_libraries(example ${ONNXRUNTIME_LIB})



