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

#ifndef __PPL_LLM_MPSC_REQUEST_SCHEDULER_H__
#define __PPL_LLM_MPSC_REQUEST_SCHEDULER_H__

#include "ppl/common/event_count.h"
#include "ppl/common/mpsc_queue.h"
#include <functional>

/** multi-producer-single-consumer request scheduler */

namespace ppl { namespace llm { namespace utils {

template<typename ReqType>
class MPSCRequestScheduler final {
public:
    MPSCRequestScheduler() {
        static_assert(std::is_base_of<ppl::common::MPSCQueue::Node, ReqType>::value,
                      "template parameter is not a derived class of ppl::common::MPSCQueue::Node");
    }

    ~MPSCRequestScheduler() {
        delete stashed_req_; // it's ok to delete nullptr

        while (true) {
            bool is_empty;
            auto node = queue_.Pop(&is_empty);
            if (!node) {
                return;
            }
            auto req = static_cast<ReqType*>(node);
            delete req;
        }
    }

    /** returns true if the queue MAY be empty before `req` is pushed */
    bool PushRequest(ReqType* req) {
        queue_.Push(req);
        uint32_t prev = size_.fetch_add(1, std::memory_order_acq_rel);
        return (prev == 0);
    }

    ReqType* TryPopRequest(const std::function<bool(const ReqType&)>& check_req_func) {
        if (stashed_req_) {
            if (!check_req_func(*stashed_req_)) {
                return nullptr;
            }

            auto req = stashed_req_;
            stashed_req_ = nullptr;
            size_.fetch_sub(1, std::memory_order_acq_rel);
            return req;
        }

        bool is_empty = true;
        ppl::common::MPSCQueue::Node* node;
        do {
            node = queue_.Pop(&is_empty);
        } while (!node && !is_empty);

        if (is_empty) {
            return nullptr;
        }

        auto req = static_cast<ReqType*>(node);
        if (check_req_func(*req)) {
            size_.fetch_sub(1, std::memory_order_acq_rel);
            return req;
        }

        stashed_req_ = req;
        return nullptr;
    }

    // approximate size
    uint32_t GetPendingSize() const {
        return size_.load(std::memory_order_relaxed);
    }

private:
    ppl::common::MPSCQueue queue_;
    std::atomic<uint32_t> size_ = {0};
    // stashed request that will be popped if `check_req_func` returns true
    ReqType* stashed_req_ = nullptr;

private:
    MPSCRequestScheduler(const MPSCRequestScheduler&) = delete;
    void operator=(const MPSCRequestScheduler&) = delete;
    MPSCRequestScheduler(MPSCRequestScheduler&&) = delete;
    void operator=(MPSCRequestScheduler&&) = delete;
};

}}} // namespace ppl::llm::utils

#endif
