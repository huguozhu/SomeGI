add_library(somegi_nanite STATIC
    nanite_types.h
    cluster_builder.h
    cluster_builder.cpp
)

target_link_libraries(somegi_nanite
    PUBLIC somegi_scene glm::glm
)
