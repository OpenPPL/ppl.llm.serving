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

#ifndef __PPL_LLM_TOKENIZER_FACTORY_H__
#define __PPL_LLM_TOKENIZER_FACTORY_H__

#include "tokenizer/tokenizer.h"
#ifdef PPL_LLM_ENABLE_HF_TOKENIZER
#include "tokenizer/tokenizer_impl_hf.h"
#endif
#include "tokenizer/tokenizer_impl_sp.h"
#include "models/llama/llama_tokenizer.h"
#include "models/internlm/internlm_tokenizer.h"
#include "models/baichuan/baichuan_tokenizer.h"
#include "models/llama3/llama3_tokenizer.h"
#include "common/resource.h"
#include "common/config.h"

#include <string>

namespace ppl { namespace llm {

class TokenizerFactory final {
public:
    static Tokenizer* Create(const std::string& model_type, const std::string& tokenizer_type,
                             const std::string& tokenizer_path, const std::string& tokenizer_config_path) {
        std::unique_ptr<TokenizerImpl> tokenizer_impl;
        if (tokenizer_type == "sentencepiece") {
            tokenizer_impl = std::make_unique<TokenizerImplSP>();
            auto rc = tokenizer_impl->Init(tokenizer_path, tokenizer_config_path);
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "sentencepiece tokenizer init failed";
                return nullptr;
            }
        #ifdef PPL_LLM_ENABLE_HF_TOKENIZER
        } else if (tokenizer_type == "huggingface") {
            tokenizer_impl = std::make_unique<TokenizerImplHF>();
            auto rc = tokenizer_impl->Init(tokenizer_path, tokenizer_config_path);
            if (rc != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "huggingface tokenizer init failed";
                return nullptr;
            }
        #endif
        } else {
            LOG(ERROR) << "not supported tokenizer: " << tokenizer_type;
            return nullptr;
        }

        std::unique_ptr<Tokenizer> tokenizer;
        if (model_type == "llama") {
            tokenizer = std::make_unique<LlamaTokenizer>(tokenizer_impl.release());
        } else if (model_type == "internlm") {
            tokenizer = std::make_unique<InternLMTokenizer>(tokenizer_impl.release());
        } else if (model_type == "baichuan") {
            tokenizer = std::make_unique<BaiChuanTokenizer>(tokenizer_impl.release());
        } else if (model_type == "llama3") {
            tokenizer = std::make_unique<Llama3Tokenizer>(tokenizer_impl.release());
        } else {
            LOG(ERROR) << "not supported model: " << model_type;
            return nullptr;
        }
        return tokenizer.release();
    }
};

}} // namespace ppl::llm

#endif
