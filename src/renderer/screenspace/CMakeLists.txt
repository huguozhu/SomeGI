add_library(somegi_renderer_screenspace OBJECT
    ssr_pass.cpp
    ssgi_pass.cpp
    gtgi_pass.cpp
)
target_include_directories(somegi_renderer_screenspace PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_screenspace PUBLIC somegi_core)
