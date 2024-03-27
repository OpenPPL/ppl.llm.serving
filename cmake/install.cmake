set(__PPLNN_CMAKE_CONFIG_FILE__ ${CMAKE_CURRENT_BINARY_DIR}/generated/pplllmserving-config.cmake)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pplllmserving-config.cmake.in
    ${__PPLNN_CMAKE_CONFIG_FILE__}
    @ONLY)
install(FILES ${__PPLNN_CMAKE_CONFIG_FILE__} DESTINATION lib/cmake/ppl)
unset(__PPLNN_CMAKE_CONFIG_FILE__)

install(TARGETS ppl_llm_static DESTINATION lib)

file(GLOB __TMP__ src/common/*.h)
install(FILES ${__TMP__} DESTINATION include/ppl/llm/common)

file(GLOB __TMP__ src/utils/*.h)
install(FILES ${__TMP__} DESTINATION include/ppl/llm/utils)

file(GLOB __TMP__ src/models/*.h)
install(FILES ${__TMP__} DESTINATION include/ppl/llm/models)

if(PPL_LLM_ENABLE_LLAMA)
    file(GLOB __TMP__ src/models/llama/*.h)
    install(FILES ${__TMP__} DESTINATION include/ppl/llm/models/llama)
endif()

if(PPL_LLM_ENABLE_GRPC_SERVING)
    file(GLOB __TMP__ src/serving/*.h)
    install(FILES ${__TMP__} DESTINATION include/ppl/llm/serving)
endif()

if(PPLNN_USE_LLM_CUDA)
    file(GLOB __TMP__ src/backends/cuda/*.h)
    install(FILES ${__TMP__} DESTINATION include/ppl/llm/backends/cuda)
endif()

unset(__TMP__)
