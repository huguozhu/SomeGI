add_library(somegi_gi STATIC
    ibl_baker.h
    ibl_baker.cpp
)

target_link_libraries(somegi_gi
    PUBLIC somegi_core somegi_scene
    PRIVATE imgui
)
