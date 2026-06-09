add_library(somegi_renderer_prt OBJECT
    prt_resources.cpp
    prt_bake_pass.cpp
)
target_include_directories(somegi_renderer_prt PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_prt PUBLIC somegi_core)
