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

#ifndef __PPL_LLM_UTILS_H__
#define __PPL_LLM_UTILS_H__

#include "ppl/common/threadpool.h"
#include "ppl/common/log.h"
#include "ppl/nn/runtime/runtime.h"

#include <functional>
#include <set>
#include <string>
#include <cstring>
#include <chrono>

namespace ppl { namespace llm { namespace utils {

void ParseTokens(const std::string& stop_tokens_str, std::set<int>* stop_tokens);

inline void DummyTaskDeleter(ppl::common::ThreadTask*) {}

template <class F, typename... TaskArgType>
ppl::common::RetCode ParallelExecute(F&& func, ppl::common::StaticThreadPool* pool, TaskArgType&&... rest_args) {
    auto n = pool->GetNumThreads();
    ppl::common::RetCode thr_rc[n];

    pool->Run([&](uint32_t nthr, uint32_t ithr) {
        thr_rc[ithr] = func(ithr, std::forward<TaskArgType>(rest_args)...);
    });
    for (uint32_t i = 0; i < n; ++i) {
        if (thr_rc[i] != ppl::common::RC_SUCCESS)
            LOG(ERROR) << "ParallelExecute task[" << i << "] failed";
        return thr_rc[i];
    }

    return ppl::common::RC_SUCCESS;
}

class TimingGuard final {
public:
    TimingGuard(uint64_t* res) {
        diff_microsec_ = res;
        begin_ = std::chrono::high_resolution_clock::now();
    }
    ~TimingGuard() {
        auto end = std::chrono::high_resolution_clock::now();
        *diff_microsec_ = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(end - begin_).count());
    }

private:
    uint64_t* diff_microsec_;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin_;
};

uint64_t HashStd(uint64_t prev, const int32_t* vec, int32_t len);

uint64_t HashCombine(uint64_t prev, const int32_t* vec, int32_t len);

bool SaveInputsOneByOne(const ppl::nn::Runtime* runtime, const std::string& save_dir, const std::string& tag);

}}} // namespace ppl::llm::utils

#endif
