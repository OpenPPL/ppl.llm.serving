if(PPLNN_USE_LLM_CUDA)
    add_executable(offline_inference tools/offline_inference.cc tools/simple_flags.cc)
    target_include_directories(offline_inference PRIVATE src)
    target_link_libraries(offline_inference PRIVATE ppl_llm_static ${NCCL_LIBRARIES})
    install(TARGETS offline_inference DESTINATION bin)

    add_executable(benchmark_prefix_cache_offline tools/benchmark_prefix_cache_offline.cc tools/simple_flags.cc)
    target_include_directories(benchmark_prefix_cache_offline PRIVATE src)
    target_link_libraries(benchmark_prefix_cache_offline PRIVATE ppl_llm_static ${NCCL_LIBRARIES})
endif()

if(PPLNN_USE_LLM_CUDA)
    if(PPL_LLM_ENABLE_GRPC_SERVING)
        add_executable(ppl_llm_server tools/llm_server.cc tools/simple_flags.cc)
        target_link_libraries(ppl_llm_server PRIVATE
            ppl_llm_static
            ppl_llm_grpc_serving_static
            ${NCCL_LIBRARIES})
        target_include_directories(ppl_llm_server PRIVATE
            ${HPCC_DEPS_DIR}/rapidjson/include
            ${NCCL_INCLUDE_DIRS})

        add_executable(client_sample tools/client_sample.cc)
        target_link_libraries(client_sample ppl_llm_grpc_proto_static grpc++)

        add_executable(client_pressure tools/client_pressure.cc)
        target_link_libraries(client_pressure ppl_llm_grpc_proto_static grpc++)

        add_executable(client_qps_measure tools/client_qps_measure.cc tools/simple_flags.cc)
        target_link_libraries(client_qps_measure pplnn_static ppl_llm_grpc_proto_static grpc++ ppl_sentencepiece_static)
        target_include_directories(client_qps_measure PUBLIC ${HPCC_DEPS_DIR}/rapidjson/include)

        add_executable(client_sample_token_in_out tools/client_sample_token_in_out.cc)
        target_link_libraries(client_sample_token_in_out ppl_llm_grpc_proto_static grpc++ ppl_sentencepiece_static)

        add_executable(client_qps_measure_token_in_out tools/client_qps_measure_token_in_out.cc tools/simple_flags.cc)
        target_link_libraries(client_qps_measure_token_in_out pplnn_static ppl_llm_grpc_proto_static grpc++)
        target_include_directories(client_qps_measure_token_in_out PUBLIC ${HPCC_DEPS_DIR}/rapidjson/include)
    endif()
endif()
