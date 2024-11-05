FetchContent_GetProperties(sentencepiece)
if(NOT sentencepiece_POPULATED)
    FetchContent_Populate(sentencepiece)
endif()

set(__PPL_LLAMA_GENERATED_DIR__ "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${__PPL_LLAMA_GENERATED_DIR__})

set(__SENTENCEPIECE_ROOT_DIR__ ${HPCC_DEPS_DIR}/sentencepiece)

if(NOT PPL_LLM_SENTENCEPIECE_PROTOBUF_LIBS)
    set(__SP_GENERATED_FILES__ "${__PPL_LLAMA_GENERATED_DIR__}/sentencepiece.pb.h;${__PPL_LLAMA_GENERATED_DIR__}/sentencepiece.pb.cc")
    set(__SPM_GENERATED_FILES__ "${__PPL_LLAMA_GENERATED_DIR__}/sentencepiece_model.pb.h;${__PPL_LLAMA_GENERATED_DIR__}/sentencepiece_model.pb.cc")

    set(__PROTO_DIR__ ${__SENTENCEPIECE_ROOT_DIR__}/src)
    add_custom_command(
        OUTPUT ${__SP_GENERATED_FILES__}
        COMMAND ${PPLNN_PROTOC_EXECUTABLE}
        ARGS --cpp_out "${__PPL_LLAMA_GENERATED_DIR__}" -I "${__PROTO_DIR__}" "${__PROTO_DIR__}/sentencepiece.proto"
        DEPENDS protoc)
    add_custom_command(
        OUTPUT ${__SPM_GENERATED_FILES__}
        COMMAND ${PPLNN_PROTOC_EXECUTABLE}
        ARGS --cpp_out "${__PPL_LLAMA_GENERATED_DIR__}" -I "${__PROTO_DIR__}" "${__PROTO_DIR__}/sentencepiece_model.proto"
        DEPENDS protoc)
    unset(__PROTO_DIR__)

    add_library(ppl_sentencepiece_pb_static STATIC
        ${__SP_GENERATED_FILES__}
        ${__SPM_GENERATED_FILES__})
    target_link_libraries(ppl_sentencepiece_pb_static PUBLIC libprotobuf)

    unset(__SPM_GENERATED_FILES__)
    unset(__SP_GENERATED_FILES__)

    set(PPL_LLM_SENTENCEPIECE_PROTOBUF_LIBS ppl_sentencepiece_pb_static)
endif()

configure_file("${__SENTENCEPIECE_ROOT_DIR__}/config.h.in" ${__PPL_LLAMA_GENERATED_DIR__}/config.h)

set(__PPL_SPM_SRCS__
    ${__SENTENCEPIECE_ROOT_DIR__}/src/bpe_model.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/common.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/normalizer.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/util.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/freelist.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/filesystem.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/init.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/sentencepiece_processor.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/word_model.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/model_factory.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/char_model.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/model_interface.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/testharness.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/unigram_model.h
    ${__SENTENCEPIECE_ROOT_DIR__}/src/bpe_model.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/char_model.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/error.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/filesystem.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/model_factory.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/model_interface.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/normalizer.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/sentencepiece_processor.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/unigram_model.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/util.cc
    ${__SENTENCEPIECE_ROOT_DIR__}/src/word_model.cc)

add_library(ppl_sentencepiece_static STATIC ${__PPL_SPM_SRCS__})
target_link_libraries(ppl_sentencepiece_static PUBLIC
    ${PPL_LLM_SENTENCEPIECE_PROTOBUF_LIBS}
    libprotobuf absl::strings absl::flags absl::flags_parse)
target_include_directories(ppl_sentencepiece_static PUBLIC
    ${__PPL_LLAMA_GENERATED_DIR__}
    ${__SENTENCEPIECE_ROOT_DIR__}
    ${__SENTENCEPIECE_ROOT_DIR__}/src)
target_compile_definitions(ppl_sentencepiece_static PRIVATE _USE_EXTERNAL_PROTOBUF)

if(PPL_LLM_INSTALL)
    install(TARGETS ${PPL_LLM_SENTENCEPIECE_PROTOBUF_LIBS} DESTINATION lib)
    install(TARGETS ppl_sentencepiece_static DESTINATION lib)
endif()

unset(__SENTENCEPIECE_ROOT_DIR__)
unset(__PPL_LLAMA_GENERATED_DIR__)
