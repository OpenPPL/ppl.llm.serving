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
    std::string model_type;
    std::string model_dir;
    std::string model_param_path;
    std::string tokenizer_path;

    int tensor_parallel_size;

    float top_p;
    int top_k;

    float max_tokens_scale;
    int max_tokens_per_request;
    int max_running_batch;
    int max_tokens_per_step;

    std::string host;
    int port;
};

struct ModelConfig final {
    int hidden_dim;
    int intermediate_dim;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int vocab_size;

    float norm_eps; // not used

    int cache_quant_bit;
    int cache_quant_group;

    int cache_layout;
    int cache_mode;

    bool dynamic_batching;
    bool auto_causal;
};

struct WorkerConfig final {
    float top_p;
    int top_k;

    int max_running_batch;
    int max_tokens_per_request;
    int max_tokens_per_step;
};
}} // namespace ppl::llm

#endif