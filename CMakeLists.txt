cmake_minimum_required(VERSION 3.14)
project(ppl.llm.serving)

option(PPL_LLM_ENABLE_GRPC_SERVING "" ON)
option(PPL_LLM_ENABLE_LLAMA "" ON)
option(PPL_LLM_ENABLE_PROFILING "" ON)
option(PPL_LLM_ENABLE_DEBUG "" OFF)
option(PPL_LLM_SERVING_BUILD_TOOLS "" ON)
option(PPL_LLM_INSTALL "" ON)

# --------------------------------------------------------------------------- #

if(PPL_LLM_ENABLE_LLAMA)
    set(CMAKE_CXX_STANDARD 17)
endif()

set(PPLNN_INSTALL ${PPL_LLM_INSTALL})

set(CMAKE_INSTALL_LIBDIR "lib")

include(cmake/deps.cmake)
string(REPLACE "-Werror=non-virtual-dtor" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# --------------------------------------------------------------------------- #

set(PPL_LLM_PROTOBUF_VERSION v23.4)

# import grpc first. for protobuf
if(PPL_LLM_ENABLE_GRPC_SERVING)
    hpcc_populate_dep(grpc)

    # use specified protobuf required by c++17
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${PPL_LLM_PROTOBUF_VERSION}
        WORKING_DIRECTORY ${grpc_SOURCE_DIR}/third_party/protobuf)

    set(PPL_LLM_PROTOC_EXECUTABLE ${grpc_BINARY_DIR}/third_party/protobuf/protoc)
elseif(PPL_LLM_ENABLE_LLAMA)
    # use specified protobuf version for c++17
    hpcc_declare_git_dep(protobuf
        https://github.com/protocolbuffers/protobuf.git
        ${PPL_LLM_PROTOBUF_VERSION})
    hpcc_populate_dep(protobuf)
    set(PPL_LLM_PROTOC_EXECUTABLE ${protobuf_BINARY_DIR}/protoc)
endif()

# --------------------------------------------------------------------------- #

if(TARGET libprotobuf)
    # generate new onnx.pb.* for pplnn
    set(__LLM_GENERATED_DIR__ ${CMAKE_CURRENT_BINARY_DIR}/generated)
    file(MAKE_DIRECTORY ${__LLM_GENERATED_DIR__})

    set(__PROTO_DIR__ ${PROJECT_SOURCE_DIR}/src/onnx)
    set(__ONNX_GENERATED_FILES__ "${__LLM_GENERATED_DIR__}/onnx.pb.h;${__LLM_GENERATED_DIR__}/onnx.pb.cc")
    add_custom_command(
        OUTPUT ${__ONNX_GENERATED_FILES__}
        COMMAND ${PPL_LLM_PROTOC_EXECUTABLE}
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
endif()

# pplnn after serving, depends on libprotobuf provided by grpc
set(PPLNN_BUILD_TOOLS OFF)
hpcc_populate_dep(pplnn)

# --------------------------------------------------------------------------- #

# serving after pplnn. depends on pplcommon
if(PPL_LLM_ENABLE_GRPC_SERVING)
    include(cmake/grpc_serving.cmake)
endif()

# --------------------------------------------------------------------------- #

include(cmake/llm.cmake)

# --------------------------------------------------------------------------- #

if(PPL_LLM_INSTALL)
    include(cmake/install.cmake)
endif()

# --------------------------------------------------------------------------- #

if(PPL_LLM_SERVING_BUILD_TOOLS)
    include(tools/CMakeLists.txt)
endif()
