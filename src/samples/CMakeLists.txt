# src/samples/CMakeLists.txt
# Compile GLSL → SPIR-V at build time
set(VERT_SPV  ${CMAKE_CURRENT_BINARY_DIR}/triangle.vert.spv)
set(FRAG_SPV  ${CMAKE_CURRENT_BINARY_DIR}/triangle.frag.spv)

add_custom_command(OUTPUT ${VERT_SPV}
    COMMAND ${GLSLANG_EXE} -V ${CMAKE_CURRENT_SOURCE_DIR}/triangle.vert -o ${VERT_SPV}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/triangle.vert
    COMMENT "glslang: triangle.vert → SPIR-V"
)
add_custom_command(OUTPUT ${FRAG_SPV}
    COMMAND ${GLSLANG_EXE} -V ${CMAKE_CURRENT_SOURCE_DIR}/triangle.frag -o ${FRAG_SPV}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/triangle.frag
    COMMENT "glslang: triangle.frag → SPIR-V"
)

# Main executable
add_executable(RhiTriangle triangle.cpp rhi_imgui.cpp ${VERT_SPV} ${FRAG_SPV})
target_include_directories(RhiTriangle PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(RhiTriangle PRIVATE somegi_rhi imgui)

# Copy SPIR-V to output dir
add_custom_command(TARGET RhiTriangle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VERT_SPV} $<TARGET_FILE_DIR:RhiTriangle>/triangle.vert.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FRAG_SPV} $<TARGET_FILE_DIR:RhiTriangle>/triangle.frag.spv
)
