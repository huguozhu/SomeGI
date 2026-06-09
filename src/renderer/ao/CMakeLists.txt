add_library(somegi_renderer_ao OBJECT
    ssao_pass.cpp
    gtao_pass.cpp
)
target_include_directories(somegi_renderer_ao PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_ao PUBLIC somegi_core)
