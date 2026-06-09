add_library(somegi_renderer_ddgi OBJECT
    ddgi_resources.cpp
    ddgi_pass.cpp
)
target_include_directories(somegi_renderer_ddgi PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_ddgi PUBLIC somegi_core)
