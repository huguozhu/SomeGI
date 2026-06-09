add_library(somegi_renderer_sdfgi OBJECT
    sdfgi_resources.cpp
    sdfgi_pass.cpp
)
target_include_directories(somegi_renderer_sdfgi PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_sdfgi PUBLIC somegi_core)
