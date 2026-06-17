# src/renderer/fg/CMakeLists.txt
add_library(somegi_renderer_fg OBJECT
    fg_graph.cpp
    fg_compiler.cpp
    fg_executor.cpp
    fg_debug.cpp
)
target_include_directories(somegi_renderer_fg PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_fg PUBLIC somegi_core imgui)
