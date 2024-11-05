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

#include "config.h"

#include "ppl/common/retcode.h"
#include "ppl/common/log.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include <fstream>
#include <vector>
#include <stdint.h>

namespace ppl { namespace llm {

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
    LOG(INFO) << "model_config.num_heads: " << model_config->num_heads;

    it = document.FindMember("num_kv_heads");
    if (it == document.MemberEnd()) {
        model_config->num_kv_heads = model_config->num_heads;
    } else {
        model_config->num_kv_heads = it->value.GetInt();
    }
    LOG(INFO) << "model_config.num_kv_heads: " << model_config->num_kv_heads;

    it = document.FindMember("num_layers");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [num_layers] failed";
        return false;
    }
    model_config->num_layers = it->value.GetInt();
    LOG(INFO) << "model_config.num_layers: " << model_config->num_layers;

    it = document.FindMember("hidden_dim");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [hidden_dim] failed";
        return false;
    }
    model_config->hidden_dim = it->value.GetInt();
    LOG(INFO) << "model_config.hidden_dim: " << model_config->hidden_dim;

    it = document.FindMember("intermediate_dim");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [intermediate_dim] failed";
        return false;
    }
    model_config->intermediate_dim = it->value.GetInt();
    LOG(INFO) << "model_config.intermediate_dim: " << model_config->intermediate_dim;

    it = document.FindMember("vocab_size");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [vocab_size] failed";
        return false;
    }
    model_config->vocab_size = it->value.GetInt();
    LOG(INFO) << "model_config.vocab_size: " << model_config->vocab_size;

    it = document.FindMember("cache_quant_bit");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_quant_bit] failed";
        return false;
    }
    model_config->cache_quant_bit = it->value.GetInt();
    LOG(INFO) << "model_config.cache_quant_bit: " << model_config->cache_quant_bit;

    it = document.FindMember("cache_quant_group");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_quant_group] failed";
        return false;
    }
    model_config->cache_quant_group = it->value.GetInt();
    LOG(INFO) << "model_config.cache_quant_group: " << model_config->cache_quant_group;

    it = document.FindMember("cache_layout");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_layout] failed";
        return false;
    }
    model_config->cache_layout = it->value.GetInt();
    LOG(INFO) << "model_config.cache_layout: " << model_config->cache_layout;

    it = document.FindMember("cache_mode");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [cache_mode] failed";
        return false;
    }
    model_config->cache_mode = it->value.GetInt();
    LOG(INFO) << "model_config.cache_mode: " << model_config->cache_mode;

    if (model_config->cache_mode == 1) {
        it = document.FindMember("page_size");
        if (it == document.MemberEnd() && model_config->cache_mode == 1) {
            LOG(ERROR) << "find key [page_size] failed";
            return false;
        }
        model_config->page_size = it->value.GetInt();
        LOG(INFO) << "model_config.page_size: " << model_config->page_size;
    }

    it = document.FindMember("dynamic_batching");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [dynamic_batching] failed";
        return false;
    }
    model_config->dynamic_batching = it->value.GetBool();
    LOG(INFO) << "model_config.dynamic_batching: " << model_config->dynamic_batching;

    it = document.FindMember("auto_causal");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [auto_causal] failed";
        return false;
    }
    model_config->auto_causal = it->value.GetBool();
    LOG(INFO) << "model_config.auto_causal: " << model_config->auto_causal;

    return true;
}

}} // namespace ppl::llm