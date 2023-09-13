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

#include "ppl/common/retcode.h"
#include "ppl/common/threadpool.h"
#include "ppl/common/barrier.h"
#include "ppl/common/log.h"
#include <chrono>
#include <vector>

namespace ppl { namespace llm { namespace utils {

inline void DummyTaskDeleter(ppl::common::ThreadTask*) {}

/**
   @brief TaskType constraints:
     - derived from ppl::common::JoinableThreadTask
     - has a constructor with parameters (uint32_t id, TaskArgType...)
     - has a member function GetRetCode()
 */
template <typename TaskType, typename... TaskArgType>
ppl::common::RetCode ParallelExecute(ppl::common::StaticThreadPool* pool, TaskArgType&&... rest_args) {
    auto n = pool->GetNumThreads();
    ppl::common::Barrier finish_barrier;
    ppl::common::RetCode thr_rc[n];
    finish_barrier.Reset(n + 1);

    pool->RunAsync([&](uint32_t nthr, uint32_t ithr) {
        auto task = TaskType(ithr, std::forward<TaskArgType>(rest_args)...);
        thr_rc[ithr] = task.Process();
        finish_barrier.Wait();
    });

    finish_barrier.Wait();
    for (uint32_t i = 0; i < n; ++i) {
        if (thr_rc[i] != ppl::common::RC_SUCCESS)
            LOG(ERROR) << "ParallelExecute task[" << i << "] failed";
        return thr_rc[i];
    }

    return ppl::common::RC_SUCCESS;
}

class TimingGuard final {
public:
    TimingGuard(double* res) {
        diff_microsec_ = res;
        begin_ = std::chrono::high_resolution_clock::now();
    }
    ~TimingGuard() {
        auto end = std::chrono::high_resolution_clock::now();
        *diff_microsec_ = double(std::chrono::duration_cast<std::chrono::microseconds>(end - begin_).count()) / 1000.0;
    }

private:
    double* diff_microsec_;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin_;
};

}}} // namespace ppl::llm::utils

#endif
