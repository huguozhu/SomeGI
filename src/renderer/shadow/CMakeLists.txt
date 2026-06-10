add_library(somegi_renderer_shadow OBJECT
    shadow_pass.cpp
)
target_include_directories(somegi_renderer_shadow PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_shadow PUBLIC somegi_core)
