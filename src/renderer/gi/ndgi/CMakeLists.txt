add_library(somegi_renderer_ndgi OBJECT
    ndgi_resources.cpp
    ndgi_pass.cpp
)
target_include_directories(somegi_renderer_ndgi PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_ndgi PUBLIC somegi_core)
