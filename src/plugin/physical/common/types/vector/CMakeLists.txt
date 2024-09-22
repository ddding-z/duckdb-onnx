if(${EXIT_TIME_DESTRUCTORS_WARNING})
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -Wno-exit-time-destructors")
endif()

add_library_unity(
  plugin_duckdb_types_vector
  OBJECT
  selection_vector.cpp
  validity_mask.cpp
  vector_buffer.cpp
  vector.cpp
  vector_cache.cpp
  vector_constants.cpp
  )
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:plugin_duckdb_types_vector>
    PARENT_SCOPE)
