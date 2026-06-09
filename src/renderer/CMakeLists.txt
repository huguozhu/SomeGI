add_subdirectory(core)
add_subdirectory(ao)
add_subdirectory(screenspace)
add_subdirectory(gi/rsm)
add_subdirectory(gi/lpv)
add_subdirectory(gi/vxgi)
add_subdirectory(gi/ddgi)
add_subdirectory(gi/ndgi)
add_subdirectory(gi/sdfgi)
add_subdirectory(gi/prt)
add_subdirectory(gi/restir)
add_subdirectory(gi/rt)
add_subdirectory(gi/lumen)

add_library(somegi_renderer STATIC
    $<TARGET_OBJECTS:somegi_renderer_core>
    $<TARGET_OBJECTS:somegi_renderer_ao>
    $<TARGET_OBJECTS:somegi_renderer_screenspace>
    $<TARGET_OBJECTS:somegi_renderer_rsm>
    $<TARGET_OBJECTS:somegi_renderer_lpv>
    $<TARGET_OBJECTS:somegi_renderer_vxgi>
    $<TARGET_OBJECTS:somegi_renderer_ddgi>
    $<TARGET_OBJECTS:somegi_renderer_ndgi>
    $<TARGET_OBJECTS:somegi_renderer_sdfgi>
    $<TARGET_OBJECTS:somegi_renderer_prt>
    $<TARGET_OBJECTS:somegi_renderer_restir>
    $<TARGET_OBJECTS:somegi_renderer_rt>
    $<TARGET_OBJECTS:somegi_renderer_lumen>
)
target_link_libraries(somegi_renderer
    PUBLIC somegi_core somegi_scene somegi_gi imgui
)
