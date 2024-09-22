if(NOT MSVC)
 set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
 # fix DEBUG macro bug
 string(REPLACE "-DDEBUG" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
 string(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# # Include the necessary modules
# include(ExternalProject)

# # Find Python interpreter
# find_package(PythonInterp 3 REQUIRED)

# if(NOT PYTHONINTERP_FOUND)
#     message(FATAL_ERROR "Python 3 interpreter not found. Please install Python 3.")
# endif()

# # Add ONNX Runtime as an external project
# ExternalProject_Add(
#     onnxruntime
#     PREFIX ${CMAKE_BINARY_DIR}/onnxruntime
#     SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime
#     CONFIGURE_COMMAND ""
#     BUILD_COMMAND ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/build.sh
#     BUILD_IN_SOURCE 1
#     INSTALL_COMMAND ""
# )

# if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
#     set(DIR_OS "MacOS")
# elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
#     set(DIR_OS "Linux")
# else()
#     message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
# endif()

# # Add the include directory and library path for ONNX Runtime
# ExternalProject_Get_Property(onnxruntime source_dir binary_dir)
# set(ONNXRUNTIME_INCLUDE_DIR ${source_dir}/include)
# # set(ONNXRUNTIME_LIB_DIR ${binary_dir}/build/Linux/Release)
# set(ONNXRUNTIME_LIB_DIR ${binary_dir}/build/${DIR_OS}/RelWithDebInfo)

# # Add the include directories and link the ONNX Runtime library
# include_directories(${ONNXRUNTIME_INCLUDE_DIR})
# link_directories(${ONNXRUNTIME_LIB_DIR})
