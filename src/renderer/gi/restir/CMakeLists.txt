add_library(somegi_renderer_restir OBJECT
    restir_resources.cpp
    restir_pass.cpp
)
target_include_directories(somegi_renderer_restir PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(somegi_renderer_restir PUBLIC somegi_core)
