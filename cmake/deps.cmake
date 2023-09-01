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

if(CMAKE_COMPILER_IS_GNUCC)
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

set(__HPCC_COMMIT__ master)

if(PPLNN_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPLNN_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    if(NOT PPLNN_DEP_HPCC_GIT)
        set(PPLNN_DEP_HPCC_GIT "https://github.com/openppl-public/hpcc.git")
    endif()
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPLNN_DEP_HPCC_GIT}
        GIT_TAG ${__HPCC_COMMIT__}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
endif()

unset(__HPCC_COMMIT__)

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

hpcc_declare_git_dep(grpc
    https://github.com/grpc/grpc.git
    v1.56.2)

# --------------------------------------------------------------------------- #

set(PPLNN_BUILD_TESTS OFF CACHE BOOL "")
set(PPLNN_BUILD_SAMPLES OFF CACHE BOOL "")

hpcc_declare_git_dep(
    ppl.nn.llm
    https://github.com/openppl-public/ppl.nn.llm.git
    master)

# --------------------------------------------------------------------------- #

hpcc_declare_git_dep(absl
    https://github.com/abseil/abseil-cpp.git
    lts_2023_01_25)

hpcc_declare_git_dep(sentencepiece
    https://github.com/openppl-public/sentencepiece.git
    ppl)
