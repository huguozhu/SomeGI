# Post-process SPIR-V to use gl_InstanceIndex directly.
# Slang generates: drawID = gl_InstanceIndex - gl_BaseInstance (= instanceID = 0)
# We need:          drawID = gl_InstanceIndex (= firstInstance + instanceID)
#
# This script uses spirv-dis + sed + spirv-as from Vulkan SDK.

function(patch_instance_index SPV_FILE)
    find_program(SPIRV_DIS spirv-dis REQUIRED)
    find_program(SPIRV_AS spirv-as REQUIRED)
    set(TMPDIR "${CMAKE_CURRENT_BINARY_DIR}/spv_patch_tmp")
    file(MAKE_DIRECTORY "${TMPDIR}")
    set(ASM "${TMPDIR}/tmp.spvasm")

    add_custom_command(
        TARGET somegi_shaders POST_BUILD
        COMMAND ${SPIRV_DIS} "${SPV_FILE}" -o "${ASM}"
        COMMAND ${CMAKE_COMMAND} -DASM="${ASM}" -P "${CMAKE_SOURCE_DIR}/cmake/patch_spv.cmake"
        COMMAND ${SPIRV_AS} "${ASM}" -o "${SPV_FILE}"
        BYPRODUCTS "${ASM}"
        COMMENT "Patching ${SPV_FILE} (InstanceIndex fix)"
        VERBATIM
    )
endfunction()
