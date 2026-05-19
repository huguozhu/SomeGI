function(compile_slang TARGET_NAME)
    set(options)
    set(oneValueArgs OUTPUT_DIR)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(SLANG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT SLANG_OUTPUT_DIR)
        set(SLANG_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
    endif()
    file(MAKE_DIRECTORY ${SLANG_OUTPUT_DIR})

    # Coarse-grained import tracking: every entry-point depends on every common
    # module. Slang has no native depfile support hooked into our build, so
    # editing any common/*.slang touches all entry points. Rebuilds are cheap
    # (sub-second per shader) and this prevents stale spv after editing imports.
    file(GLOB _common_deps CONFIGURE_DEPENDS
         "${CMAKE_SOURCE_DIR}/shaders/common/*.slang")

    set(_outputs)
    foreach(SRC ${SLANG_SOURCES})
        get_filename_component(_name ${SRC} NAME_WE)
        file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/shaders ${SRC})
        get_filename_component(_rel_dir ${_rel} DIRECTORY)
        set(_outdir "${SLANG_OUTPUT_DIR}/${_rel_dir}")
        file(MAKE_DIRECTORY ${_outdir})
        set(_out "${_outdir}/${_name}.spv")
        add_custom_command(
            OUTPUT ${_out}
            COMMAND ${SLANGC_EXE}
                ${SRC}
                -I ${CMAKE_SOURCE_DIR}/shaders/common
                -target spirv
                -profile spirv_1_5
                -o ${_out}
                -fvk-use-entrypoint-name
                -emit-spirv-directly
            DEPENDS ${SRC} ${_common_deps}
            COMMENT "slangc ${_rel}"
            VERBATIM
        )
        list(APPEND _outputs ${_out})
    endforeach()
    add_custom_target(${TARGET_NAME} DEPENDS ${_outputs})
endfunction()
