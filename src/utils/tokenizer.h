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

#ifndef __PPL_LLM_UTILS_TOKENIZER_H__
#define __PPL_LLM_UTILS_TOKENIZER_H__

#include <string>
#include <vector>

namespace ppl { namespace llm { namespace utils {

class Tokenizer {
public:
    virtual ~Tokenizer() {}
    virtual void Encode(const char* prompt, uint32_t len, std::vector<int>* token_ids) const = 0;
    virtual void Decode(int* token_ids, uint32_t len, std::string* output) const = 0;
    virtual bool IsEosId(int token_id) const = 0;
    virtual int GetEosId() const = 0;
};

}}} // namespace ppl::llm::utils

#endif