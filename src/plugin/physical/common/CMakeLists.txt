if(NOT CLANG_TIDY)
  add_subdirectory(adbc)
endif()

add_subdirectory(arrow)
add_subdirectory(file_system)
add_subdirectory(sort)
add_subdirectory(types)

if(${EXIT_TIME_DESTRUCTORS_WARNING})
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -Wno-exit-time-destructors")
endif()

add_library_unity(
  plugin_duckdb_common
  OBJECT
  allocator.cpp
  fsst.cpp
  radix_partitioning.cpp
  )
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:plugin_duckdb_common>
    PARENT_SCOPE)
