add_library(somegi_nanite STATIC
    nanite_types.h
    cluster_builder.h
    cluster_builder.cpp
    nanite_gpu.h
    nanite_gpu.cpp
)

target_link_libraries(somegi_nanite
    PUBLIC somegi_core somegi_scene glm::glm
)
