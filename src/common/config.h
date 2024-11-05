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

#ifndef __PPL_LLM_CONFIG_H__
#define __PPL_LLM_CONFIG_H__

#include <string>
#include <stdint.h>
#include <set>

namespace ppl { namespace llm {

struct ResourceConfig final {
    std::string model_type;
    std::string model_format;
    std::string model_dir;
    std::string model_param_path;
    int32_t tensor_parallel_size = 0;
    float max_tokens_scale = 0.f;
    int32_t max_running_batch = 0;
    bool enable_penalty = false;
    struct EngineConfig {
        std::string cublas_layout_hint = "default";
        bool disable_graph_fusion = false;
        bool disable_decoding_shm_mha = false;
        bool disable_decoding_inf_mha = false;
        bool disable_decoding_inf_gqa = false;
        int32_t configure_decoding_attn_split_k = 1;
        int32_t specify_decoding_attn_tpb = 0;
        std::string quant_method;
    };
    EngineConfig engine_config;
};

struct GeneratorConfig final {
    float top_p = 0.0f;
    int32_t top_k = 1;
    bool enable_penalty = false;
    int32_t max_running_batch = 0;
    int32_t max_input_tokens_per_request = 0;
    int32_t max_output_tokens_per_request = 0;
    int32_t max_total_tokens_per_request = 0;
    int32_t max_tokens_per_step = 0;
    std::set<int> stop_tokens;
    std::set<int> special_tokens;
    int max_cooldown_request = 2;
    bool enable_prefix_cache = false;
    int32_t max_prefill_batch = 0;
};

struct ModelConfig final {
    int32_t hidden_dim = 0;
    int32_t intermediate_dim = 0;
    int32_t num_layers = 0;
    int32_t num_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t vocab_size = 0;

    float norm_eps = 0.0f; // not used

    int32_t cache_quant_bit = 0;
    int32_t cache_quant_group = 0;

    int32_t cache_layout = 0;
    int32_t cache_mode = 0;
    int32_t page_size = 0;

    bool dynamic_batching = true;
    bool auto_causal = true;
};

bool ParseModelConfig(const std::string& model_param_path, ModelConfig* model_config);

}} // namespace ppl::llm

#endif