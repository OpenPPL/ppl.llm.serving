# sentencepiece after serving, libprotobuf and absl wanted
include(cmake/sentencepiece.cmake)

file(GLOB __PPL_LLM_SRC__ src/models/factory.cc src/models/llama/llama_worker.cc src/models/internlm/internlm_worker.cc)

if(PPLNN_USE_LLM_CUDA)
    file(GLOB __TMP_SRC__ src/backends/cuda/*.cc)
    list(APPEND __PPL_LLM_SRC__ ${__TMP_SRC__})
    unset(__TMP_SRC__)
endif()

if(PPL_LLM_ENABLE_PROFILING)
    add_definitions(-DPPL_LLM_ENABLE_PROFILING)
endif()

if(PPL_LLM_ENABLE_DEBUG)
    add_definitions(-DPPL_LLM_ENABLE_DEBUG)
endif()

add_library(ppl_llm_static STATIC ${__PPL_LLM_SRC__})
target_link_libraries(ppl_llm_static PUBLIC pplnn_static ppl_sentencepiece_static)
target_include_directories(ppl_llm_static PUBLIC ${PROJECT_SOURCE_DIR}/src)
if(PPL_LLM_ENABLE_PROFILING)
    target_compile_definitions(ppl_llm_static PUBLIC PPL_LLM_ENABLE_PROFILING)
endif()
if(PPL_LLM_ENABLE_DEBUG)
    target_compile_definitions(ppl_llm_static PUBLIC PPL_LLM_ENABLE_DEBUG)
endif()