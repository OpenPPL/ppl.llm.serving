if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${HPCC_DEPS_DIR})
set(FETCHCONTENT_QUIET OFF)

if(PPLNN_HOLD_DEPS)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

# --------------------------------------------------------------------------- #

if(PPL_LLM_ENABLE_LLAMA AND CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0.0)
        message(FATAL_ERROR "gcc >= 9.0.0 is required.")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 10.3.0)
        message(FATAL_ERROR "gcc 10.3.0 has known bugs. use another version >= 9.0.0.")
    endif()
endif()

# --------------------------------------------------------------------------- #

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(FATAL_ERROR "git is required.")
endif()

if(NOT PPL_LLM_DEP_HPCC_VERSION)
    set(PPL_LLM_DEP_HPCC_VERSION master)
endif()

if(PPL_LLM_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPL_LLM_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    if(NOT PPL_LLM_DEP_HPCC_GIT)
        set(PPL_LLM_DEP_HPCC_GIT "https://github.com/OpenPPL/hpcc.git")
    endif()
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPL_LLM_DEP_HPCC_GIT}
        GIT_TAG ${PPL_LLM_DEP_HPCC_VERSION}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
endif()

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# ------------------------------------------------------------------------- #

set(gRPC_BUILD_TESTS OFF CACHE BOOL "")
set(gRPC_BUILD_CSHARP_EXT OFF CACHE BOOL "")
set(gRPC_BUILD_GRPC_CSHARP_PLUGIN OFF CACHE BOOL "")
set(gRPC_BUILD_GRPC_NODE_PLUGIN OFF CACHE BOOL "")
set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN OFF CACHE BOOL "")
set(gRPC_BUILD_GRPC_PHP_PLUGIN OFF CACHE BOOL "")
set(gRPC_BUILD_GRPC_PYTHON_PLUGIN OFF CACHE BOOL "")
set(gRPC_BUILD_GRPC_RUBY_PLUGIN OFF CACHE BOOL "")
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "")
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "required by protobuf")

# --------------------------------------------------------------------------- #

if(PPL_LLM_DEP_GRPC_PKG)
    hpcc_declare_pkg_dep(grpc
        ${PPL_LLM_DEP_GRPC_PKG})
else()
    if(NOT PPL_LLM_DEP_GRPC_GIT)
        set(PPL_LLM_DEP_GRPC_GIT "https://github.com/grpc/grpc.git")
    endif()
    hpcc_declare_git_dep_depth1(grpc
        ${PPL_LLM_DEP_GRPC_GIT}
        v1.56.2)
endif()

# --------------------------------------------------------------------------- #

set(PPLNN_BUILD_TESTS OFF CACHE BOOL "")
set(PPLNN_BUILD_SAMPLES OFF CACHE BOOL "")

if(NOT PPL_LLM_DEP_PPLNN_VERSION)
    set(PPL_LLM_DEP_PPLNN_VERSION llm_v2)
endif()

if(PPL_LLM_DEP_PPLNN_PKG)
    hpcc_declare_pkg_dep(pplnn
        ${PPL_LLM_DEP_PPLNN_PKG})
else()
    if(NOT PPL_LLM_DEP_PPLNN_GIT)
        set(PPL_LLM_DEP_PPLNN_GIT "https://github.com/OpenPPL/ppl.nn.git")
    endif()
hpcc_declare_git_dep(pplnn
    ${PPL_LLM_DEP_PPLNN_GIT}
    ${PPL_LLM_DEP_PPLNN_VERSION})
endif()

# --------------------------------------------------------------------------- #

if(PPL_LLM_DEP_ABSL_PKG)
    hpcc_declare_pkg_dep(absl
        ${PPL_LLM_DEP_ABSL_PKG})
else()
    if(NOT PPL_LLM_DEP_ABSL_GIT)
        set(PPL_LLM_DEP_ABSL_GIT "https://github.com/abseil/abseil-cpp.git")
    endif()
    hpcc_declare_git_dep_depth1(absl
        ${PPL_LLM_DEP_ABSL_GIT}
        lts_2023_01_25)
endif()

# --------------------------------------------------------------------------- #

if(PPL_LLM_DEP_SENTENCEPIECE_PKG)
    hpcc_declare_pkg_dep(sentencepiece
        ${PPL_LLM_DEP_SENTENCEPIECE_PKG})
else()
    if(NOT PPL_LLM_DEP_SENTENCEPIECE_GIT)
        set(PPL_LLM_DEP_SENTENCEPIECE_GIT "https://github.com/OpenPPL/sentencepiece.git")
    endif()
    hpcc_declare_git_dep_depth1(sentencepiece
        ${PPL_LLM_DEP_SENTENCEPIECE_GIT}
        ppl)
endif()

# --------------------------------------------------------------------------- #

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(ENABLE_PUSH OFF CACHE BOOL "")
set(ENABLE_COMPRESSION OFF CACHE BOOL "")
set(ENABLE_TESTING OFF CACHE BOOL "")
set(GENERATE_PKGCONFIG OFF CACHE BOOL "")
set(OVERRIDE_CXX_STANDARD_FLAGS OFF CACHE BOOL "")

# --------------------------------------------------------------------------- #

if(PPL_LLM_DEP_PROMETHEUS_PKG)
    hpcc_declare_pkg_dep(prometheus
        ${PPL_LLM_DEP_PROMETHEUS_PKG})
else()
    if(NOT PPL_LLM_DEP_PROMETHEUS_GIT)
        set(PPL_LLM_DEP_PROMETHEUS_GIT "https://github.com/jupp0r/prometheus-cpp.git")
    endif()
    hpcc_declare_git_dep_depth1(prometheus
        ${PPL_LLM_DEP_PROMETHEUS_GIT}
        v1.2.4)
endif()

# --------------------------------------------------------------------------- #

if(PPL_LLM_DEP_XXHASH_PKG)
    hpcc_declare_pkg_dep(xxhash
        ${PPL_LLM_DEP_XXHASH_PKG})
else()
    if(NOT PPL_LLM_DEP_XXHASH_GIT)
        set(PPL_LLM_DEP_XXHASH_GIT "https://github.com/Cyan4973/xxHash.git")
    endif()
    hpcc_declare_git_dep_depth1(xxhash
        ${PPL_LLM_DEP_XXHASH_GIT}
        v0.8.2)
endif()

# --------------------------------------------------------------------------- #

if(PPL_LLM_DEP_TOKENIZER_CPP_PKG)
    hpcc_declare_pkg_dep(tokenizer_cpp
        ${PPL_LLM_DEP_TOKENIZER_CPP_PKG})
else()
    if(NOT PPL_LLM_DEP_TOKENIZER_CPP_GIT)
        set(PPL_LLM_DEP_TOKENIZER_CPP_GIT "https://github.com/OpenPPL/tokenizers-cpp.git")
    endif()
    hpcc_declare_git_dep_depth1(tokenizer_cpp
        ${PPL_LLM_DEP_TOKENIZER_CPP_GIT}
        llm_v2)
endif()
