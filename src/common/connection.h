#ifndef __PPL_LLM_CONNECTION_H__
#define __PPL_LLM_CONNECTION_H__

#include "response.h"

namespace ppl { namespace llm {

class Connection {
public:
    virtual ~Connection() {}
    virtual void Send(const std::vector<Response>&) = 0;
    virtual void NotifyFailure(uint64_t id) = 0;
};
}} // namespace ppl::llm

#endif
