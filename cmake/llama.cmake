# sentencepiece after serving, libprotobuf and absl wanted
include(cmake/sentencepiece.cmake)

file(GLOB __PPL_LLAMA_SRC__ src/models/llama/llama_worker.cc)

if(PPLNN_USE_LLM_CUDA)
    file(GLOB __TMP_SRC__ src/sampler/cuda/*.cc)
    list(APPEND __PPL_LLAMA_SRC__ ${__TMP_SRC__})
    unset(__TMP_SRC__)
endif()

if(PPL_LLM_ENABLE_PROFILING)
    add_definitions(-DPPL_LLM_ENABLE_PROFILING)
endif()

if(PPL_LLM_ENABLE_DEBUG)
    add_definitions(-DPPL_LLM_ENABLE_DEBUG)
endif()

find_package(OpenMP REQUIRED)

add_library(ppl_llm_llama_static STATIC ${__PPL_LLAMA_SRC__})
target_link_libraries(ppl_llm_llama_static PUBLIC pplnn_static ppl_sentencepiece_static OpenMP::OpenMP_CXX)
target_include_directories(ppl_llm_llama_static PUBLIC ${PROJECT_SOURCE_DIR}/src)
if(PPL_LLM_ENABLE_PROFILING)
    target_compile_definitions(ppl_llm_llama_static PUBLIC PPL_LLM_ENABLE_PROFILING)
endif()
if(PPL_LLM_ENABLE_DEBUG)
    target_compile_definitions(ppl_llm_llama_static PUBLIC PPL_LLM_ENABLE_DEBUG)
endif()

if(PPL_LLM_ENABLE_GRPC_SERVING)
    add_executable(ppl_llama_server src/models/llama/llama_server.cc)
    target_link_libraries(ppl_llama_server PRIVATE
        ppl_llm_llama_static
        ppl_llm_grpc_serving_static
        ${NCCL_LIBRARIES})
    target_include_directories(ppl_llama_server PRIVATE
        ${HPCC_DEPS_DIR}/rapidjson/include
        ${NCCL_INCLUDE_DIRS})
endif()
