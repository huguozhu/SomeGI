add_library(somegi_scene STATIC
    scene.h
    camera.h
    camera.cpp
    upload.h
    upload.cpp
    gltf_loader.h
    gltf_loader.cpp
    env_loader.h
    env_loader.cpp
    scene_gpu.h
    scene_gpu.cpp
)

target_link_libraries(somegi_scene
    PUBLIC somegi_core
    PRIVATE cgltf stb_headers
)
