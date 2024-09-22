add_subdirectory(column)
add_subdirectory(row)
add_subdirectory(row_operations)
add_subdirectory(vector)
add_subdirectory(vector_operations)

if(${EXIT_TIME_DESTRUCTORS_WARNING})
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -Wno-exit-time-destructors")
endif()

add_library_unity(
  plugin_duckdb_types
  OBJECT
  batched_data_collection.cpp
  chunk_collection.cpp
  conflict_manager.cpp
  conflict_info.cpp
  data_chunk.cpp
  hyperloglog.cpp
  list_segment.cpp
  string_heap.cpp
  )
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:plugin_duckdb_types>
    PARENT_SCOPE)
