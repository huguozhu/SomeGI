add_library(somegi_renderer_lpv OBJECT
    lpv_grid.cpp
    lpv_inject_pass.cpp
    lpv_propagate_pass.cpp
)
target_include_directories(somegi_renderer_lpv PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_lpv PUBLIC somegi_core)
