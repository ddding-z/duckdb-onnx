add_subdirectory(common)
add_subdirectory(execution)
add_subdirectory(storage)
add_subdirectory(parallel)
add_subdirectory(transaction)

set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES}
    PARENT_SCOPE)