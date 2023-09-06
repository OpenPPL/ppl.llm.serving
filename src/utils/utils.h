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
ppl::common::RetCode ParallelExecute(ppl::common::ThreadPool* workers, uint32_t n, TaskArgType&&... rest_args) {
    auto task_list = (TaskType*)malloc(n * sizeof(TaskType));
    if (!task_list) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    for (uint32_t i = 0; i < n; ++i) {
        new (task_list + i) TaskType(i, std::forward<TaskArgType>(rest_args)...);
        workers[i].AddTask(std::shared_ptr<ppl::common::ThreadTask>(task_list + i, DummyTaskDeleter));
    }

    uint32_t ok_count = 0;
    for (uint32_t i = 0; i < n; ++i) {
        task_list[i].Join();
        ok_count += (task_list[i].GetRetCode() == ppl::common::RC_SUCCESS);
        task_list[i].~TaskType();
    }

    free(task_list);

    return ((ok_count == n) ? ppl::common::RC_SUCCESS : ppl::common::RC_OTHER_ERROR);
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
