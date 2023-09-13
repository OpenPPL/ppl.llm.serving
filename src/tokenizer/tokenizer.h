#ifndef __PPL_LLM_TOKENIZER_H__
#define __PPL_LLM_TOKENIZER_H__

#include <string>
#include <vector>

namespace ppl { namespace llm {

class Tokenizer {
public:
    virtual void Encode(const char* prompt, uint32_t len, std::vector<int>* token_ids) const = 0;
    virtual void Decode(int* token_ids, uint32_t len, std::string* output) const = 0;
    virtual bool IsEosId(int token_id) const = 0;
};

}} // namespace ppl::llm

#endif