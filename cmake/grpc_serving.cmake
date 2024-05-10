set(__LLM_SERVING_GENERATED_DIR__ "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${__LLM_SERVING_GENERATED_DIR__})

set(__PROTO_DIR__ ${PROJECT_SOURCE_DIR}/src/serving/grpc/proto)
set(PROTOC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/grpc-build/third_party/protobuf/protoc")

# ----- cannot disable zlib tests and examples ----- #

if(TARGET minigzip)
    set_target_properties(minigzip PROPERTIES LINK_LIBRARIES z)
endif()
if(TARGET minigzip64)
    set_target_properties(minigzip64 PROPERTIES LINK_LIBRARIES z)
endif()
if(TARGET example)
    set_target_properties(example PROPERTIES LINK_LIBRARIES z)
endif()
if(TARGET example64)
    set_target_properties(example64 PROPERTIES LINK_LIBRARIES z)
endif()

# ----- grpc serving pb files ----- #

set(__LLM_GENERATED_FILES__ "${__LLM_SERVING_GENERATED_DIR__}/llm.pb.cc;${__LLM_SERVING_GENERATED_DIR__}/llm.pb.h;${__LLM_SERVING_GENERATED_DIR__}/llm.grpc.pb.cc;${__LLM_SERVING_GENERATED_DIR__}/llm.grpc.pb.h")

set(GRPC_CPP_PLUGIN_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/grpc-build/grpc_cpp_plugin")
add_custom_command(
    OUTPUT ${__LLM_GENERATED_FILES__}
    COMMAND ${PROTOC_EXECUTABLE}
    ARGS --grpc_out "${__LLM_SERVING_GENERATED_DIR__}" --cpp_out "${__LLM_SERVING_GENERATED_DIR__}"
    -I "${__PROTO_DIR__}" --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${__PROTO_DIR__}/llm.proto"
    DEPENDS protoc grpc_cpp_plugin ${__PROTO_DIR__}/llm.proto)

add_library(ppl_llm_grpc_proto_static STATIC ${__LLM_GENERATED_FILES__})
target_link_libraries(ppl_llm_grpc_proto_static PUBLIC libprotobuf grpc)
target_include_directories(ppl_llm_grpc_proto_static PUBLIC
    ${HPCC_DEPS_DIR}/grpc/include
    ${__LLM_SERVING_GENERATED_DIR__})

# ----- #

file(GLOB __SRC__ src/serving/grpc/*.cc)
add_library(ppl_llm_grpc_serving_static STATIC ${__SRC__})
target_link_libraries(ppl_llm_grpc_serving_static PUBLIC ppl_llm_static ppl_llm_grpc_proto_static grpc++ pplcommon_static pthread)
target_include_directories(ppl_llm_grpc_serving_static PUBLIC src)
