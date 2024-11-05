# sentencepiece after serving, libprotobuf and absl wanted

file(GLOB __PPL_LLM_SRC__
    src/common/*.cc
    src/engine/*.cc
    src/generator/*.cc
    src/utils/*.cc
)

if(PPLNN_USE_LLM_CUDA)
    if(HPCC_ENABLE_SANITIZE_OPTIONS)
        message(FATAL_ERROR "`HPCC_ENABLE_SANITIZE_OPTIONS` can not be used with nccl now.")
    endif()
    file(GLOB __TMP_SRC__ src/backends/cuda/*.cc)
    list(APPEND __PPL_LLM_SRC__ ${__TMP_SRC__})
    unset(__TMP_SRC__)
endif()

include(cmake/sentencepiece.cmake)

add_library(ppl_llm_static STATIC ${__PPL_LLM_SRC__})
target_link_libraries(ppl_llm_static PUBLIC
    pplnn_static
    ppl_sentencepiece_static)
if (PPL_LLM_ENABLE_HF_TOKENIZER)
    target_link_libraries(ppl_llm_static PUBLIC tokenizers_cpp tokenizers_c)
    target_compile_definitions(ppl_llm_static PUBLIC PPL_LLM_ENABLE_HF_TOKENIZER)
endif()

target_compile_options(ppl_llm_static PUBLIC ${HPCC_SANITIZE_COMPILE_OPTIONS})
target_include_directories(ppl_llm_static PUBLIC ${HPCC_DEPS_DIR}/rapidjson/include)
target_link_options(ppl_llm_static PUBLIC ${HPCC_SANITIZE_LINK_OPTIONS})
install(TARGETS ppl_llm_static DESTINATION lib)

if(PPL_LLM_ENABLE_PROFILING)
    target_compile_definitions(ppl_llm_static PUBLIC PPL_LLM_ENABLE_PROFILING)
endif()
if(PPL_LLM_ENABLE_DEBUG)
    target_compile_definitions(ppl_llm_static PUBLIC PPL_LLM_ENABLE_DEBUG)
endif()
if(PPLNN_CUDA_ENABLE_NCCL)
    target_compile_definitions(ppl_llm_static PUBLIC PPLNN_CUDA_ENABLE_NCCL)
endif()
if(PPLNN_USE_LLM_CUDA)
    target_compile_definitions(ppl_llm_static PUBLIC PPLNN_USE_LLM_CUDA)
endif()
if (PPL_LLM_SERVING_SYNC_DECODE)
    target_compile_definitions(ppl_llm_static PUBLIC PPL_LLM_SERVING_SYNC_DECODE)
endif()
