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

#ifndef __PPL_LLM_BAICHUAN_TOKENIZER_H__
#define __PPL_LLM_BAICHUAN_TOKENIZER_H__

#include "utils/tokenizer.h"

#include "absl/strings/string_view.h"
#include "ppl/nn/common/logger.h"
#include "sentencepiece_processor.h"


namespace ppl { namespace llm { namespace baichuan {

class BaiChuanTokenizer final : public utils::Tokenizer {
public:
    ppl::common::RetCode Init(const std::string& path) {
        auto tokenizer_status = sp_processor_.Load(path);
        if (!tokenizer_status.ok()) {
            LOG(ERROR) << tokenizer_status.ToString();
            return ppl::common::RC_OTHER_ERROR;
        }
        LOG(INFO) << "VOCAB_SIZE: " << sp_processor_.GetPieceSize() << "; BOS ID: " << sp_processor_.bos_id()
                  << "; EOS ID: " << sp_processor_.eos_id() << "; PAD ID: " << sp_processor_.pad_id();
        return ppl::common::RC_SUCCESS;
    }

    void Encode(const char* prompt, uint32_t len, std::vector<int>* token_ids) const override {
        sp_processor_.Encode(absl::string_view(prompt, len), token_ids);
    }

    void Decode(int* token_ids, uint32_t len, std::string* output) const override {
        sp_processor_.Decode(token_ids, len, output);
        if (IsNewWord(token_ids[0])) {
            output->insert(0, " ");
        }
    }

    bool IsEosId(int token_id) const override {
        return token_id == sp_processor_.eos_id();
    }

private:
    bool IsNewWord(int token_id) const {
        return sp_processor_.IdToPiece(token_id).substr(0, 3) == "▁";
    }

private:
    sentencepiece::SentencePieceProcessor sp_processor_;
};

}}} // namespace ppl::llm::baichuan

#endif