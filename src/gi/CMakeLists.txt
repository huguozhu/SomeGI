add_library(somegi_gi STATIC
    gi_technique.h
    ibl_baker.h
    ibl_baker.cpp
    ibl_technique.h
    ibl_technique.cpp
)

target_link_libraries(somegi_gi
    PUBLIC somegi_core somegi_scene
    PRIVATE imgui
)
