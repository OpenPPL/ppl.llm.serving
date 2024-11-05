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

#ifndef __PPL_LLM_LLAMA_TOKENIZER_H__
#define __PPL_LLM_LLAMA_TOKENIZER_H__

#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_impl.h"
#include "absl/strings/string_view.h"
#include "ppl/nn/common/logger.h"

namespace ppl { namespace llm {

class LlamaTokenizer final : public Tokenizer {
public:
    LlamaTokenizer(TokenizerImpl* impl) {
        impl_ = std::unique_ptr<TokenizerImpl>(impl);
    }
    ~LlamaTokenizer() {}

    void Encode(const char* prompt, uint32_t len, std::vector<int>* token_ids) const override {
        impl_->Encode(prompt, len, token_ids);
        token_ids->insert(token_ids->begin(), impl_->GetBosId());
    }

    void Decode(int* token_ids, uint32_t len, std::string* output) const override {
        impl_->Decode(token_ids, len, output);
    }

    int GetBosId() const override {
        return impl_->GetBosId();
    }

    int GetEosId() const override {
        return impl_->GetEosId();
    }

private:
    std::unique_ptr<TokenizerImpl> impl_;
};

}} // namespace ppl::llm

#endif
