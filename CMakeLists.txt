cmake_minimum_required(VERSION 3.14)
project(ppl.llm.serving)

option(PPL_LLM_ENABLE_LLAMA "" ON)
option(PPL_LLM_ENABLE_PROFILING "" OFF)
option(PPL_LLM_ENABLE_DEBUG "" OFF)
option(PPL_LLM_SERVING_BUILD_TOOLS "" ON)
option(PPL_LLM_INSTALL "" ON)
option(PPL_LLM_ENABLE_GRPC_SERVING "" ON)
option(PPL_LLM_SERVING_SYNC_DECODE "" OFF)

option(PPL_LLM_ENABLE_HF_TOKENIZER "" ON)
option(PPL_LLM_ENABLE_TEST "" OFF)

# --------------------------------------------------------------------------- #
if(HPCC)
    include(hpcc-common)
endif()

if(PPL_LLM_ENABLE_LLAMA)
    set(CMAKE_CXX_STANDARD 17)
endif()

set(PPLNN_INSTALL ${PPL_LLM_INSTALL})

set(CMAKE_INSTALL_LIBDIR "lib")

include(cmake/deps.cmake)

# --------------------------------------------------------------------------- #

if (PPL_LLM_ENABLE_HF_TOKENIZER)
    hpcc_populate_dep(tokenizer_cpp)
endif()

# import grpc first. for protobuf
# set protobuf version before importing ppl.nn
set(PPLNN_DEP_PROTOBUF_VERSION v23.4)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "")
set(protobuf_BUILD_TESTS OFF CACHE BOOL "disable protobuf tests")

hpcc_populate_dep(grpc)

# use specified protobuf required by c++17
find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${PPLNN_DEP_PROTOBUF_VERSION}
    WORKING_DIRECTORY ${grpc_SOURCE_DIR}/third_party/protobuf)

set(PPLNN_PROTOC_EXECUTABLE ${grpc_BINARY_DIR}/third_party/protobuf/protoc)

# --------------------------------------------------------------------------- #

# generate new onnx.pb.* for pplnn
set(__LLM_GENERATED_DIR__ ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${__LLM_GENERATED_DIR__})

set(__PROTO_DIR__ ${PROJECT_SOURCE_DIR}/src/onnx)
set(__ONNX_GENERATED_FILES__ "${__LLM_GENERATED_DIR__}/onnx.pb.h;${__LLM_GENERATED_DIR__}/onnx.pb.cc")
add_custom_command(
    OUTPUT ${__ONNX_GENERATED_FILES__}
    COMMAND ${PPLNN_PROTOC_EXECUTABLE}
    ARGS --cpp_out ${__LLM_GENERATED_DIR__} -I ${__PROTO_DIR__}
    ${__PROTO_DIR__}/onnx.proto
    DEPENDS protoc ${__PROTO_DIR__}/onnx.proto)
add_library(pplnn_onnx_generated_static STATIC ${__ONNX_GENERATED_FILES__})
target_link_libraries(pplnn_onnx_generated_static PUBLIC libprotobuf)
target_include_directories(pplnn_onnx_generated_static PUBLIC ${__LLM_GENERATED_DIR__})
set(PPLNN_ONNX_GENERATED_LIBS pplnn_onnx_generated_static)

unset(__ONNX_GENERATED_FILES__)
unset(__PROTO_DIR__)
unset(__LLM_GENERATED_DIR__)

# pplnn after serving, depends on libprotobuf provided by grpc
hpcc_populate_dep(pplnn)

# --------------------------------------------------------------------------- #

# serving after pplnn. depends on pplcommon
if(PPL_LLM_ENABLE_GRPC_SERVING)
    include(cmake/grpc_serving.cmake)
endif()

# --------------------------------------------------------------------------- #
if(PPL_LLM_ENABLE_LLAMA)
    include(cmake/llm.cmake)
endif()

# --------------------------------------------------------------------------- #

if(PPL_LLM_INSTALL)
    include(cmake/install.cmake)
endif()

# --------------------------------------------------------------------------- #

if(PPL_LLM_SERVING_BUILD_TOOLS)
    include(tools/CMakeLists.txt)
endif()

if(PPL_LLM_ENABLE_TEST)
    add_subdirectory(test)
endif()
