add_library(somegi_renderer_rt OBJECT
    scene_rt_as.cpp
    rt_gi_pass.cpp
)
target_include_directories(somegi_renderer_rt PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_rt PUBLIC somegi_core)
