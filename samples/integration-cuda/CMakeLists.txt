cmake_minimum_required(VERSION 3.10)
project(ppl-llm-integration-cuda)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# cuda env initializations
find_package(CUDA REQUIRED)
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
endif()
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
enable_language(CUDA)

# import ppl llm serving(without grpc)
get_filename_component(pplllmserving_DIR "${CMAKE_CURRENT_LIST_DIR}/../../ppl-build/install/lib/cmake/ppl" ABSOLUTE)
# optional: disable unused devices to avoid linking extra deps
find_package(pplllmserving REQUIRED)

# ------------------- #

# sample target
add_executable(ppl-llm-integration-cuda main.cc)
target_include_directories(ppl-llm-integration-cuda PRIVATE
    ${PPL_LLM_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # imported from `enable_language(CUDA)`
target_link_directories(ppl-llm-integration-cuda PRIVATE
    ${PPL_LLM_LINK_DIRS}
    ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES}) # imported from `enable_language(CUDA)`
target_link_libraries(ppl-llm-integration-cuda PRIVATE ${PPL_LLM_LIBRARIES})
