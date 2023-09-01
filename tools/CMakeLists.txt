if(PPL_LLM_ENABLE_GRPC_SERVING)
    add_executable(client_sample tools/client_sample.cc)
    target_link_libraries(client_sample ppl_llm_grpc_proto_static grpc++)

    add_executable(client_pressure tools/client_pressure.cc)
    target_link_libraries(client_pressure ppl_llm_grpc_proto_static grpc++)

    add_executable(client_qps_measure tools/client_qps_measure.cc)
    target_link_libraries(client_qps_measure pplnn_static ppl_llm_grpc_proto_static grpc++ ppl_sentencepiece_static)
    target_include_directories(client_qps_measure PUBLIC ${HPCC_DEPS_DIR}/rapidjson/include)
endif()
