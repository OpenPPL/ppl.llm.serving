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

#include "factory.h"
#include "llama/llama_worker.h"
#include "llama/llama_tokenizer.h"
#include "internlm/internlm_tokenizer.h"
#include "baichuan/baichuan_tokenizer.h"

#include "ppl/common/log.h"

#include <memory>

using namespace ppl::common;

namespace ppl { namespace llm {

RequestProcessor* ModelFactory::Create(const std::string& model_type, const Resource& resource,
                                       const ModelConfig& mconfig, const WorkerConfig& wconfig,
                                       Connection* c) {
    if (model_type == "llama" || model_type == "baichuan" || model_type == "internlm") {
        auto* llama_worker = new llama::LLaMAWorker(resource, mconfig, wconfig, c);
        auto rc = llama_worker->Init();
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "llama_worker init failed: " << GetRetCodeStr(rc);
            return nullptr;
        }
        LOG(INFO) << "Init llama worker successed";
        return llama_worker;
    } else {
        LOG(ERROR) << "not supported model: " << model_type;
        return nullptr;
    }
}

utils::Tokenizer* TokenizerFactory::Create(const std::string& model_type, const std::string& tokenizer_path) {
    if (model_type == "llama") {
        auto* llama_tokenizer = new llama::LlamaTokenizer();
        llama_tokenizer->Init(tokenizer_path);
        return llama_tokenizer;
    } else if (model_type == "internlm") {
        auto* internlm_tokenizer = new internlm::InternLmTokenizer();
        internlm_tokenizer->Init(tokenizer_path);
        return internlm_tokenizer;
    } else if (model_type == "baichuan") {
        auto* internlm_tokenizer = new baichuan::BaiChuanTokenizer();
        internlm_tokenizer->Init(tokenizer_path);
        return internlm_tokenizer;
    } else {
        LOG(ERROR) << "not supported model: " << model_type;
        return nullptr;
    }
}

}} // namespace ppl::llm
