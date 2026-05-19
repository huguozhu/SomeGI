add_executable(SomeGI
    main.cpp
    app.h
    app.cpp
)
target_link_libraries(SomeGI PRIVATE somegi_core somegi_scene somegi_gi somegi_renderer)
target_compile_definitions(SomeGI PRIVATE
    SOMEGI_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
    SOMEGI_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
    SOMEGI_PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)
if(TARGET somegi_shaders)
    add_dependencies(SomeGI somegi_shaders)
endif()
set_target_properties(SomeGI PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
