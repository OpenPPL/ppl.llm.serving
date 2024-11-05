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

#ifndef __PPL_LLM_TOKENIZER_IMPL_HF_H__
#define __PPL_LLM_TOKENIZER_IMPL_HF_H__

#include "tokenizer_impl.h"
#ifdef PPL_LLM_ENABLE_HF_TOKENIZER
#include "tokenizers_cpp.h"
#endif
#include "ppl/nn/common/logger.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include <iostream>
#include <string>
#include <fstream>

namespace ppl { namespace llm {

static bool LoadBytesFromFile(const std::string& path, std::string* data) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        LOG(ERROR) << "Cannot open " << path;
        return false;
    }
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data->resize(size);
    fs.read(data->data(), size);
    return true;
}

static bool ParseTokenizerConfig(const std::string& conifg_path, std::string* bos_token, std::string* eos_token) {
    std::ifstream ifs(conifg_path);
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document document;
    if (document.ParseStream(isw) == false) {
        LOG(ERROR) << "ParseStream failed";
        return false;
    }
    document.ParseStream(isw);

    auto it = document.FindMember("bos_token");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [bos_token] failed";
        return false;
    }
    *bos_token = it->value.GetString();

    it = document.FindMember("eos_token");
    if (it == document.MemberEnd()) {
        LOG(ERROR) << "find key [eos_token] failed";
        return false;
    }
    *eos_token = it->value.GetString();

    return true;
}

class TokenizerImplHF final : public TokenizerImpl {
public:
    ~TokenizerImplHF() {}
    ppl::common::RetCode Init(const std::string& path, const std::string& conifg_path = "") override {
        if (conifg_path.empty()) {
            LOG(ERROR) << "No config file for HuggingFace Tokenizer";
            return ppl::common::RC_OTHER_ERROR;
        }
        std::string bos_token, eos_token;
        if (!ParseTokenizerConfig(conifg_path, &bos_token, &eos_token)) {
            LOG(ERROR) << "ParseTokenizerConfig failed";
            return ppl::common::RC_OTHER_ERROR;
        }

        std::string blob;
        if (!LoadBytesFromFile(path, &blob)) {
            LOG(ERROR) << "LoadBytesFromFile failed";
            return ppl::common::RC_OTHER_ERROR;
        }

        hf_processor_ = ::tokenizers::Tokenizer::FromBlobJSON(blob);
        if (!hf_processor_) {
            LOG(ERROR) << "Init HuggingFace Tokenizer failed";
            return ppl::common::RC_OTHER_ERROR;
        }
        LOG(INFO) << "VOCAB_SIZE: " << hf_processor_->GetVocabSize();
        LOG(INFO) << "bos_token: " << bos_token;
        LOG(INFO) << "eos_token: " << eos_token;
        bos_id_ = hf_processor_->TokenToId(bos_token);
        if (bos_id_ == -1) {
            LOG(ERROR) << "illegal bos token, bos_id_ is -1";
            return ppl::common::RC_OTHER_ERROR;
        }
        eos_id_ = hf_processor_->TokenToId(eos_token);
        if (eos_id_ == -1) {
            LOG(ERROR) << "illegal eos token, eos_id_ is -1";
            return ppl::common::RC_OTHER_ERROR;
        }
        return ppl::common::RC_SUCCESS;
    }

    void Encode(const char* prompt, uint32_t len, std::vector<int>* token_ids) const override {
        hf_processor_->Encode(prompt, len, token_ids);
    }

    void Decode(int* token_ids, uint32_t len, std::string* output) const override {
        hf_processor_->Decode(token_ids, len, output);
    }

    int GetBosId() const override {
        return bos_id_;
    }

    int GetEosId() const override {
        return eos_id_;
    }

private:
    std::unique_ptr<tokenizers::Tokenizer> hf_processor_;
    int bos_id_;
    int eos_id_;
};

}} // namespace ppl::llm

#endif