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
Define_string_opt("--special_tokens", g_flag_special_tokens, "", "special tokens")
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

class LocalConnection final : public Connection {
public:
    LocalConnection() {
        pthread_mutex_init(&finish_lock_, nullptr);
        pthread_cond_init(&finish_signal_, nullptr);
    }

    ~LocalConnection() {
        pthread_cond_destroy(&finish_signal_);
        pthread_mutex_destroy(&finish_lock_);
    }

    void OnTokenize(uint64_t, const vector<int>&) override {}

    void OnProfiling(const std::shared_ptr<WorkerProfiler>& worker_profiler) override {
        PrintProfiler(*worker_profiler);
    }

    void Send(const vector<Response>& batched_rsp) override {
        pthread_mutex_lock(&finish_lock_);
        for (const auto& rsp : batched_rsp) {
            auto& rsp_str = tid_rsp_map_->emplace(rsp.id, std::string()).first->second;
            rsp_str += rsp.generated;
            if (rsp.finish_flag != FinishFlag::NOT_FINISHED) {
                ++count_;
            }
        }

        if (count_ >= wanted_) {
            pthread_cond_signal(&finish_signal_);
        }
        pthread_mutex_unlock(&finish_lock_);
    }

    void NotifyFailure(uint64_t, RetCode, const string&) override {
        pthread_mutex_lock(&finish_lock_);
        ++count_;
        if (count_ >= wanted_) {
            pthread_cond_signal(&finish_signal_);
        }
        pthread_mutex_unlock(&finish_lock_);
    }

    void Wait() {
        pthread_mutex_lock(&finish_lock_);
        while (count_ < wanted_) {
            pthread_cond_wait(&finish_signal_, &finish_lock_);
        }
        count_ = 0;
        pthread_mutex_unlock(&finish_lock_);
    }

    void SetTidRspMap(std::unordered_map<uint64_t, std::string>* tid_rsp_map) {
        tid_rsp_map_ = tid_rsp_map;
    }

    void SetWanted(uint32_t wanted) {
        wanted_ = wanted;
    }

private:
    std::unordered_map<uint64_t, std::string>* tid_rsp_map_;
    uint32_t wanted_;
    uint32_t count_ = 0;

    pthread_mutex_t finish_lock_;
    pthread_cond_t finish_signal_;
};

int main(int argc, char* argv[]) {
    const vector<string> prompts = {
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    };

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

    vector<std::shared_ptr<Request>> request_list;
    for (size_t i = 0; i < prompts.size(); ++i) {
        request_list.push_back(std::make_shared<Request>(i, prompts[i], 1.0, 8 + i));
    }
    unordered_map<uint64_t, string> tid_rsp_map;
    LocalConnection local_conn;
    local_conn.SetWanted(request_list.size());
    local_conn.SetTidRspMap(&tid_rsp_map);

    auto llm_generator = std::make_unique<LLMGenerator>(resource, generator_config, model_config, &local_conn);
    rc = llm_generator->Init();
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "llm_generator init failed: " << GetRetCodeStr(rc);
    }

    LOG(INFO) << "before generate";

    uint64_t generate_time;
    {
        ppl::llm::utils::TimingGuard __timing__(&generate_time);
        for (auto req : request_list) {
            llm_generator->Process(req);
        }
        local_conn.Wait();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (size_t i = 0; i < tid_rsp_map.size(); i++) {
        const std::string& prompt = request_list[i]->prompt;
        const std::string& answer = tid_rsp_map[request_list[i]->id];

        std::stringstream ss;
        ss << "Prompt: " << prompt << std::endl;
        ss << "Answer:" << answer;

        LOG(INFO) << "\n" << ss.str();
    }

    std::cout << "generation time: " << double(generate_time) / 1e3 << "ms" << std::endl;
    llm_generator.reset();

    return 0;
}
