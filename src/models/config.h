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

namespace ppl { namespace llm {

struct ServerConfig final {
    std::string model_type = "";
    std::string model_dir = "";
    std::string model_param_path = "";
    std::string tokenizer_path = "";
    bool use_pmx = false;

    int tensor_parallel_size = 0;

    float top_p = 0.0f;
    int top_k = 0;

    std::string quant_method = "";

    float max_tokens_scale = 0.0f;
    int max_tokens_per_request = 0;
    int max_running_batch = 0;
    int max_tokens_per_step = 0;

    std::string host = "";
    int port = 0;
};

struct ModelConfig final {
    int hidden_dim = 0;
    int intermediate_dim = 0;
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int vocab_size = 0;

    float norm_eps = 0.0f; // not used

    int cache_quant_bit = 0;
    int cache_quant_group = 0;

    int cache_layout = 0;
    int cache_mode = 0;

    bool dynamic_batching = false;
    bool auto_causal = false;
};

struct WorkerConfig final {
    float top_p = 0.0f;
    int top_k = 0;

    int max_running_batch = 0;
    int max_tokens_per_request = 0;
    int max_tokens_per_step = 0;
};

}} // namespace ppl::llm

#endif