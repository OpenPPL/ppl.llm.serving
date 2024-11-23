// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "simple_flags.h"
#include "backends/cuda/resource_manager.h"
#include "common/config.h"
#include "common/connection.h"
#include "tokenizer/tokenizer_factory.h"
#include "serving/grpc/grpc_server.h"
#include "utils/utils.h"
#include "generator/llm_generator.h"
#include "ppl/common/log.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <thread>

using namespace std;
using namespace ppl::llm;
using namespace ppl::common;
using namespace ppl::nn;

/* ------------------------------------------------------------------------- */

Define_bool_opt("--help", g_flag_help, false, "show these help information");
Define_bool_opt("--version", g_flag_version, false, "show version info");

Define_string_opt("--model-type", g_flag_model_type, "llama", "");
Define_string_opt("--model-format", g_flag_model_format, "onnx", "");
Define_string_opt("--model-dir", g_flag_model_dir, "", "");
Define_string_opt("--model-param-path", g_flag_model_param_path, "", "");
Define_int32_opt("--tensor-parallel-size", g_flag_tensor_parallel_size, 1, "");
Define_bool_opt("--enable-penalty", g_flag_enable_penalty, false, "whether enable penalty");
Define_float_opt("--max-tokens-scale", g_flag_max_tokens_scale, 0.94, "");
Define_float_opt("--top-p", g_flag_top_p, 0.0, "");
Define_int32_opt("--top-k", g_flag_top_k, 1, "");
Define_int32_opt("--max-input-tokens-per-request", g_flag_max_input_tokens_per_request, 4096, "");
Define_int32_opt("--max-output-tokens-per-request", g_flag_max_output_tokens_per_request, 4096, "");
Define_int32_opt("--max-total-tokens-per-request", g_flag_max_total_tokens_per_request, 8192, "");
Define_int32_opt("--max-running-batch", g_flag_max_running_batch, 1024, "");
Define_int32_opt("--max-tokens-per-step", g_flag_max_tokens_per_step, 8192, "");
Define_int32_opt("--max-cooldown-request", g_flag_max_cooldown_reqeust, 2,
                 "when gpu mem is full, wait for number for task to be completed");
Define_string_opt("--quant-method", g_flag_quant_method, "none", "");
Define_string_opt("--cublas-layout-hint", g_flag_cublas_layout_hint, "default",
                  "matrix layout hint for cublas(currently only effect int8 gemm), only accept "
                  "\"default\", \"ampere\". "
                  "default: \"default\"");
Define_bool_opt("--disable-decoding-shm-mha", g_flag_disable_decoding_shm_mha, false,
                "disable shared memory decoding attention algorithm");
Define_bool_opt("--disable-decoding-inf-mha", g_flag_disable_decoding_inf_mha, false,
                "disable infinity decoding attention algorithm");
Define_bool_opt("--disable-decoding-inf-gqa", g_flag_disable_decoding_inf_gqa, false,
                "disable infinity grouped query decoding attention algorithm");
Define_int32_opt("--configure-decoding-attn-split-k", g_flag_configure_decoding_attn_split_k, 1,
                 "configuring split-k decoding attention algorithm, "
                 "accepted values: always-on(2)/heuristic(1)/off(0),"
                 "default is heuristic(1)");
Define_int32_opt("--specify-decoding-attn-tpb", g_flag_specify_decoding_attn_tpb, 0,
                 "specify decoding attention kernel threads per block, "
                 "accepted values: 512/256/heuristic(0),"
                 "default is heuristic(0)");
Define_bool_opt("--disable-graph-fusion", g_flag_disable_graph_fusion, false, "disable graph kernel fusion rules");
Define_string_opt("--stop-tokens", g_flag_stop_tokens, "", "stop tokens list");
Define_string_opt("--special-tokens", g_flag_special_tokens, "", "special tokens")
Define_string_opt("--tokenizer-path", g_flag_tokenizer_path, "", "");
Define_string_opt("--tokenizer-type", g_flag_tokenizer_type, "sentencepiece", "sentencepiece/hugging face");
Define_string_opt("--tokenizer-config-path", g_flag_tokenizer_config_path, "", "/path/to/utils/tokenizer_config.json");
Define_string_opt("--host", g_flag_host, "127.0.0.1", "");
Define_int32_opt("--port", g_flag_port, 10086, "");
Define_int32_opt("--monitor-port", g_flag_monitor_port, 23333, "");
Define_int32_opt("--control-port", g_flag_control_port, 12345, "");
Define_bool_opt("--enable-prefix-cache", g_flag_enable_prefix_cache, false, "is enable prefix cache");
Define_int32_opt("--max-prefill-batch", g_flag_max_prefill_batch, 64, "max prefill batches per step");
Define_bool_opt("--enable-profiling", g_flag_enable_profiling, false, "print profiling message");

/* ------------------------------------------------------------------------- */

static bool CheckInputArgs() {
#define PrintArg(var) \
    LOG(INFO) << #var << ": "  << g_flag_##var;

    if (g_flag_enable_prefix_cache) {
        g_flag_max_prefill_batch = 1;
    }
    PrintArg(model_type);
    PrintArg(model_format);
    PrintArg(model_dir);
    PrintArg(model_param_path);
    PrintArg(tensor_parallel_size);
    PrintArg(enable_penalty);
    PrintArg(max_tokens_scale);
    PrintArg(top_p);
    PrintArg(top_k);
    PrintArg(max_input_tokens_per_request);
    PrintArg(max_output_tokens_per_request);
    PrintArg(max_total_tokens_per_request);
    PrintArg(max_running_batch);
    PrintArg(max_tokens_per_step);
    PrintArg(max_cooldown_reqeust);
    PrintArg(quant_method);
    PrintArg(cublas_layout_hint);
    PrintArg(disable_decoding_shm_mha);
    PrintArg(disable_decoding_inf_mha);
    PrintArg(disable_decoding_inf_gqa);
    PrintArg(configure_decoding_attn_split_k);
    PrintArg(specify_decoding_attn_tpb);
    PrintArg(disable_graph_fusion);
    PrintArg(stop_tokens);
    PrintArg(special_tokens);
    PrintArg(tokenizer_path);
    PrintArg(tokenizer_type);
    PrintArg(tokenizer_config_path);
    PrintArg(host);
    PrintArg(port);
    PrintArg(monitor_port);
    PrintArg(control_port);
    PrintArg(enable_prefix_cache);
    PrintArg(max_prefill_batch);
    PrintArg(enable_profiling);

    if (g_flag_tensor_parallel_size <= 0 || (g_flag_tensor_parallel_size & (g_flag_tensor_parallel_size - 1)) != 0) {
        LOG(ERROR) << "tensor_parallel_size must be power of 2, which is " << g_flag_tensor_parallel_size;
        return false;
    }

    if (g_flag_top_p < 0 || g_flag_top_p > 1) {
        LOG(ERROR) << "top_p must be in range [0, 1], which is " << g_flag_top_p;
        return false;
    }

    if (g_flag_top_k <= 0) {
        LOG(ERROR) << "top_k must be greater than 0, which is " << g_flag_top_k;
        return false;
    }

    if (g_flag_max_tokens_scale <= 0 || g_flag_max_tokens_scale >= 1) {
        LOG(ERROR) << "max_tokens_scale must be in range (0, 1), which is " << g_flag_max_tokens_scale;
        return false;
    }

    if (g_flag_max_input_tokens_per_request <= 0) {
        LOG(ERROR) << "max_input_tokens_per_request must be greater than 0, which is "
                   << g_flag_max_input_tokens_per_request;
        return false;
    }

    if (g_flag_max_output_tokens_per_request <= 0) {
        LOG(ERROR) << "max_output_tokens_per_request must be greater than 0, which is "
                   << g_flag_max_output_tokens_per_request;
        return false;
    }

    if (g_flag_max_total_tokens_per_request <= 0) {
        LOG(ERROR) << "max_total_tokens_per_request must be greater than 0, which is "
                   << g_flag_max_total_tokens_per_request;
        return false;
    }

    if (g_flag_max_running_batch <= 0) {
        LOG(ERROR) << "max_running_batch must be greater than 0, which is " << g_flag_max_running_batch;
        return false;
    }

    if (g_flag_max_tokens_per_step <= 0) {
        LOG(ERROR) << "max_tokens_per_step must be greater than 0, which is " << g_flag_max_tokens_per_step;
        return false;
    }

    if (g_flag_port <= 0 || g_flag_port > 65535) {
        LOG(ERROR) << "port must be in range [1, 65535], which is " << g_flag_port;
        return false;
    }

    return true;

#undef PPLPrintArg
}

static bool SetResourceConfig(ResourceConfig* resource_config) {
    resource_config->model_type = g_flag_model_type;
    resource_config->model_format = g_flag_model_format;
    resource_config->model_dir = g_flag_model_dir;
    resource_config->model_param_path = g_flag_model_param_path;
    resource_config->tensor_parallel_size = g_flag_tensor_parallel_size;
    resource_config->max_tokens_scale = g_flag_max_tokens_scale;
    resource_config->max_running_batch = g_flag_max_running_batch;
    resource_config->enable_penalty = g_flag_enable_penalty;
    resource_config->engine_config.cublas_layout_hint = g_flag_cublas_layout_hint;
    resource_config->engine_config.disable_graph_fusion = g_flag_disable_graph_fusion;

    resource_config->engine_config.disable_decoding_shm_mha = g_flag_disable_decoding_shm_mha;
    resource_config->engine_config.disable_decoding_inf_mha = g_flag_disable_decoding_inf_mha;
    resource_config->engine_config.disable_decoding_inf_gqa = g_flag_disable_decoding_inf_gqa;
    resource_config->engine_config.configure_decoding_attn_split_k = g_flag_configure_decoding_attn_split_k;

    resource_config->engine_config.specify_decoding_attn_tpb = g_flag_specify_decoding_attn_tpb;
    resource_config->engine_config.quant_method = g_flag_quant_method;
    return true;
}

static bool SetGeneratorConfig(GeneratorConfig* generator_config) {
    generator_config->top_p = g_flag_top_p;
    generator_config->top_k = g_flag_top_k;
    generator_config->enable_penalty = g_flag_enable_penalty;
    generator_config->max_running_batch = g_flag_max_running_batch;
    generator_config->max_input_tokens_per_request = g_flag_max_input_tokens_per_request;
    generator_config->max_output_tokens_per_request = g_flag_max_output_tokens_per_request;
    generator_config->max_total_tokens_per_request = g_flag_max_total_tokens_per_request;
    generator_config->max_tokens_per_step = g_flag_max_tokens_per_step;
    ppl::llm::utils::ParseTokens(g_flag_stop_tokens, &generator_config->stop_tokens);
    ppl::llm::utils::ParseTokens(g_flag_special_tokens, &generator_config->special_tokens);
    generator_config->max_cooldown_request = g_flag_max_cooldown_reqeust;
    generator_config->enable_prefix_cache = g_flag_enable_prefix_cache;
    generator_config->max_prefill_batch = g_flag_max_prefill_batch;
    generator_config->enable_profiling = g_flag_enable_profiling;
    return true;
}

int main(int argc, char* argv[]) {
    simple_flags::parse_args(argc, argv);
    if (!simple_flags::get_unknown_flags().empty()) {
        string content;
        for (auto it : simple_flags::get_unknown_flags()) {
            content += "'" + it + "', ";
        }
        content.resize(content.size() - 2); // remove last ', '
        content.append(".");
        LOG(ERROR) << "unknown option(s): " << content.c_str();
        return -1;
    }

    if (g_flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    if (!CheckInputArgs()) {
        LOG(ERROR) << "Check input args failed";
        return -1;
    }

    ResourceConfig resource_config;
    if (!SetResourceConfig(&resource_config)) {
        LOG(ERROR) << "SetResourceConfig error";
        return -1;
    }

    GeneratorConfig generator_config;
    if (!SetGeneratorConfig(&generator_config)) {
        LOG(ERROR) << "SetGeneratorConfig error";
        return -1;
    }

    ModelConfig model_config;
    if (!ParseModelConfig(g_flag_model_param_path, &model_config)) {
        LOG(ERROR) << "PaseModelConfig failed, model_param_path: " << g_flag_model_param_path;
        return -1;
    }
    LOG(INFO) << "Parse model model_config successed";

    auto tokenizer = unique_ptr<Tokenizer>(TokenizerFactory::Create(
        resource_config.model_type, g_flag_tokenizer_type, g_flag_tokenizer_path, g_flag_tokenizer_config_path));
    if (!tokenizer) {
        LOG(ERROR) << "create tokenizer failed";
        return -1;
    }

    // init nccl, cuda engine, kv cache, kv scale manager
    cuda::CudaResourceManager resource_manager;
    auto rc = resource_manager.Init(model_config, resource_config);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "init CudaResourceManager failed: " << GetRetCodeStr(rc);
        return -1;
    }

    Resource resource;
    resource.tensor_parallel_size = resource_config.tensor_parallel_size;
    resource.kv_cache_max_tokens = resource_manager.kv_cache_max_tokens;
    resource.items = resource_manager.items;
    resource.post_processor = resource_manager.post_processor.get();
    resource.device_worker_pool_ = &resource_manager.device_worker_pool_;
    resource.tokenizer = tokenizer.get();

    GRPCConnection conn;

    auto llm_generator = std::make_unique<LLMGenerator>(resource, generator_config, model_config, &conn);
    rc = llm_generator->Init();
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "llm_generator init failed: " << GetRetCodeStr(rc);
    }

    auto on_disconnected_func = [worker = llm_generator.get()](uint64_t uid) {
        worker->ClearTask(uid);
    };

    GRPCServer svr(&conn, on_disconnected_func);
    auto listen_addr = g_flag_host + ":" + std::to_string(g_flag_port);
    rc = svr.Init(listen_addr);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "GRPCConnection init failed.";
        return -1;
    }

    LOG(INFO) << "listening on [" << listen_addr << "]";

    svr.Loop(llm_generator.get());
    llm_generator.reset();
    return 0;
}
