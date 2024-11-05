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

#ifndef __PPL_LLM_TOKENIZER_IMPL_SP_H__
#define __PPL_LLM_TOKENIZER_IMPL_SP_H__

#include "tokenizer_impl.h"
#include "ppl/nn/common/logger.h"
#include "sentencepiece_processor.h"

#include <iostream>
#include <string>
#include <fstream>

namespace ppl { namespace llm {

class TokenizerImplSP final : public TokenizerImpl {
public:
    ~TokenizerImplSP() {}
    ppl::common::RetCode Init(const std::string& path, const std::string& conifg_path = "") override {
        sp_processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
        auto tokenizer_status = sp_processor_->Load(path);
        if (!tokenizer_status.ok()) {
            LOG(ERROR) << tokenizer_status.ToString();
            return ppl::common::RC_OTHER_ERROR;
        }
        bos_id_ = sp_processor_->bos_id();
        eos_id_ = sp_processor_->eos_id();

        LOG(INFO) << "VOCAB_SIZE: " << sp_processor_->GetPieceSize() << "; BOS ID: " << bos_id_
                  << "; EOS ID: " << eos_id_ << "; PAD ID: " << sp_processor_->pad_id();
        return ppl::common::RC_SUCCESS;
    }

    void Encode(const char* prompt, uint32_t len, std::vector<int>* token_ids) const override {
        sp_processor_->Encode(absl::string_view(prompt, len), token_ids);
    }

    void Decode(int* token_ids, uint32_t len, std::string* output) const override {
        sp_processor_->Decode(token_ids, len, output);
        if (len == 1 && sp_processor_->IdToPiece(token_ids[0]).substr(0, 3) == space_symbol_ && !output->empty() &&
            output->at(0) != ' ') {
            output->insert(0, " ");
        }
    }

    int GetBosId() const override {
        return bos_id_;
    }

    int GetEosId() const override {
        return eos_id_;
    }

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor_;
    int bos_id_;
    int eos_id_;
    const std::string space_symbol_ = "\xe2\x96\x81";
};

}} // namespace ppl::llm

#endif