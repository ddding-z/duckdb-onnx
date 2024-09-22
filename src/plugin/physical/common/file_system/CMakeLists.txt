if(${EXIT_TIME_DESTRUCTORS_WARNING})
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -Wno-exit-time-destructors")
endif()

add_library_unity(
  plugin_duckdb_file_system
  OBJECT
  compressed_file_system.cpp
  file_buffer.cpp
  file_system.cpp
  filename_pattern.cpp
  gzip_file_system.cpp
  hive_partitioning.cpp
  http_state.cpp
  pipe_file_system.cpp
  local_file_system.cpp
  multi_file_reader.cpp
  virtual_file_system.cpp
  )
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:plugin_duckdb_file_system>
    PARENT_SCOPE)
