function(add_header_only_library NAME INCLUDE_DIR)
    add_library(${NAME} INTERFACE)
    target_include_directories(${NAME} INTERFACE ${INCLUDE_DIR})
endfunction()
