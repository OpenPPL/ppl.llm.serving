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

#ifndef __PPL_LLM_UTILS_CONFIG_UTILS_H__
#define __PPL_LLM_UTILS_CONFIG_UTILS_H__

#include "models/config.h"

#include "ppl/common/retcode.h"
#include "ppl/common/log.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include <fstream>
#include <string>
#include <vector>

namespace ppl { namespace llm { namespace utils {

bool ParseServerConfig(const std::string& config_file, ServerConfig* server_config) {
    std::ifstream ifs(config_file);
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document json_reader;
    json_reader.ParseStream(isw);
    if (json_reader.HasParseError()) {
        LOG(ERROR) << "Parse Json Error, server config file: " << config_file;
        return false;
    }

    auto it = json_reader.FindMember("model_type");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [model_type] failed";
        return false;
    }
    server_config->model_type = it->value.GetString();

    it = json_reader.FindMember("model_dir");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [model_dir] failed";
        return false;
    }
    server_config->model_dir = it->value.GetString();

    it = json_reader.FindMember("model_param_path");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [model_param_path] failed";
        return false;
    }
    server_config->model_param_path = it->value.GetString();

    it = json_reader.FindMember("tokenizer_path");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [tokenizer_path] failed";
        return false;
    }
    server_config->tokenizer_path = it->value.GetString();

    it = json_reader.FindMember("tensor_parallel_size");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [tensor_parallel_size] failed";
        return false;
    }
    server_config->tensor_parallel_size = it->value.GetInt();

    it = json_reader.FindMember("top_p");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [top_p] failed";
        return false;
    }
    server_config->top_p = std::min(std::max(it->value.GetFloat(), 0.0f), 1.0f);

    it = json_reader.FindMember("top_k");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [top_k] failed";
        return false;
    }
    server_config->top_k = std::max(it->value.GetInt(), 1);

    it = json_reader.FindMember("max_tokens_scale");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [max_tokens_scale] failed";
        return false;
    }
    server_config->max_tokens_scale = std::max(it->value.GetFloat(), 0.1f);

    it = json_reader.FindMember("max_tokens_per_request");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [max_tokens_per_request] failed";
        return false;
    }
    server_config->max_tokens_per_request = std::max(it->value.GetInt(), 1);

    it = json_reader.FindMember("max_running_batch");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [max_running_batch] failed";
        return false;
    }
    server_config->max_running_batch = std::max(it->value.GetInt(), 1);

    it = json_reader.FindMember("max_tokens_per_step");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [max_tokens_per_step] failed";
    }
    server_config->max_tokens_per_step = std::max(it->value.GetInt(), 1);

    it = json_reader.FindMember("host");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [host] failed";
        return false;
    }
    server_config->host = it->value.GetString();

    it = json_reader.FindMember("port");
    if (it == json_reader.MemberEnd()) {
        LOG(ERROR) << "find key [port] failed";
        return false;
    }
    server_config->port = it->value.GetInt();

    LOG(INFO) << "server_config.host: " << server_config->host;
    LOG(INFO) << "server_config.port: " << server_config->port;

    LOG(INFO) << "server_config.model_type: " << server_config->model_type;
    LOG(INFO) << "server_config.model_dir: " << server_config->model_dir;
    LOG(INFO) << "server_config.model_param_path: " << server_config->model_param_path;
    LOG(INFO) << "server_config.tokenizer_path: " << server_config->tokenizer_path;

    LOG(INFO) << "server_config.top_k: " << server_config->top_k;
    LOG(INFO) << "server_config.top_p: " << server_config->top_p;

    LOG(INFO) << "server_config.tensor_parallel_size: " << server_config->tensor_parallel_size;
    LOG(INFO) << "server_config.max_tokens_scale: " << server_config->max_tokens_scale;
    LOG(INFO) << "server_config.max_tokens_per_request: " << server_config->max_tokens_per_request;
    LOG(INFO) << "server_config.max_running_batch: " << server_config->max_running_batch;
    LOG(INFO) << "server_config.max_tokens_per_step: " << server_config->max_tokens_per_step;

    return true;
}

bool ParseModelConfig(const std::string& model_param_path, ModelConfig* model_config) {
    std::ifstream ifs(model_param_path);
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document document;
    if (document.ParseStream(isw) == false) {
        LOG(ERROR) << "ParseStream failed";
        return false;
    }
    document.ParseStream(isw);

    auto it = document.FindMember("num_heads");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [num_heads] failed";
        return false;
    }
    model_config->num_heads = it->value.GetInt();

    it = document.FindMember("num_kv_heads");
    if (it == document.MemberEnd()) {
        model_config->num_kv_heads = model_config->num_heads;
    } else {
        model_config->num_kv_heads = it->value.GetInt();
    }

    it = document.FindMember("num_layers");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [num_layers] failed";
        return false;
    }
    model_config->num_layers = it->value.GetInt();

    it = document.FindMember("hidden_dim");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [hidden_dim] failed";
        return false;
    }
    model_config->hidden_dim = it->value.GetInt();

    it = document.FindMember("intermediate_dim");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [intermediate_dim] failed";
        return false;
    }
    model_config->intermediate_dim = it->value.GetInt();

    it = document.FindMember("vocab_size");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [vocab_size] failed";
        return false;
    }
    model_config->vocab_size = it->value.GetInt();

    it = document.FindMember("cache_quant_bit");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_quant_bit] failed";
        return false;
    }
    model_config->cache_quant_bit = it->value.GetInt();

    it = document.FindMember("cache_quant_group");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_quant_group] failed";
        return false;
    }
    model_config->cache_quant_group = it->value.GetInt();

    it = document.FindMember("cache_layout");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_layout] failed";
        return false;
    }
    model_config->cache_layout = it->value.GetInt();

    it = document.FindMember("cache_mode");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_mode] failed";
        return false;
    }
    model_config->cache_mode = it->value.GetInt();

    it = document.FindMember("dynamic_batching");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [dynamic_batching] failed";
        return false;
    }
    model_config->dynamic_batching = it->value.GetBool();

    it = document.FindMember("auto_causal");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [auto_causal] failed";
        return false;
    }
    model_config->auto_causal = it->value.GetBool();

    LOG(INFO) << "model_config.num_layers: " << model_config->num_layers;
    LOG(INFO) << "model_config.num_heads: " << model_config->num_heads;
    LOG(INFO) << "model_config.num_kv_heads: " << model_config->num_kv_heads;
    LOG(INFO) << "model_config.hidden_dim: " << model_config->hidden_dim;
    LOG(INFO) << "model_config.intermediate_dim: " << model_config->intermediate_dim;
    LOG(INFO) << "model_config.vocab_size: " << model_config->vocab_size;

    LOG(INFO) << "model_config.cache_quant_bit: " << model_config->cache_quant_bit;
    LOG(INFO) << "model_config.cache_quant_group: " << model_config->cache_quant_group;
    LOG(INFO) << "model_config.cache_layout: " << model_config->cache_layout;
    LOG(INFO) << "model_config.cache_mode: " << model_config->cache_mode;

    LOG(INFO) << "model_config.dynamic_batching: " << model_config->dynamic_batching;
    LOG(INFO) << "model_config.auto_causal: " << model_config->auto_causal;

    return true;
}

}}} // namespace ppl::llm::utils

#endif