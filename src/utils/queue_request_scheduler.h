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

#ifndef __PPL_LLM_QUEUE_REQUEST_SCHEDULER_H__
#define __PPL_LLM_QUEUE_REQUEST_SCHEDULER_H__

#include <queue>
#include <functional>
#include <memory>
#include <pthread.h>

namespace ppl { namespace llm { namespace utils {

template <typename ReqType>
class QueueRequestScheduler final {
public:
    QueueRequestScheduler() {
        consumer_queue_index_ = 0;
        pthread_mutex_init(&switch_queue_lock_, nullptr);
    }

    ~QueueRequestScheduler() {
        pthread_mutex_destroy(&switch_queue_lock_);
    }

    void PushRequest(const std::shared_ptr<ReqType>& req) {
        pthread_mutex_lock(&switch_queue_lock_);
        queues_[1 - consumer_queue_index_].push(req);
        pthread_mutex_unlock(&switch_queue_lock_);
    }

    void PushRequests(const std::function<std::shared_ptr<ReqType>()>& f) {
        std::shared_ptr<ReqType> req;
        pthread_mutex_lock(&switch_queue_lock_);
        auto producer_idx = 1 - consumer_queue_index_;
        while ((req = f())) {
            queues_[producer_idx].push(req);
        }
        pthread_mutex_unlock(&switch_queue_lock_);
    }

    std::shared_ptr<ReqType> TryPopRequest(const std::function<bool(const ReqType&)>& check_req_func) {
        std::shared_ptr<ReqType> req;

        if (queues_[consumer_queue_index_].empty()) {
            pthread_mutex_lock(&switch_queue_lock_);
            auto next_idx = 1 - consumer_queue_index_;
            if (!queues_[next_idx].empty()) {
                consumer_queue_index_ = next_idx;
                req = queues_[next_idx].front();
                if (check_req_func(*req)) {
                    queues_[next_idx].pop();
                } else {
                    req.reset();
                }
            }
            pthread_mutex_unlock(&switch_queue_lock_);
        } else {
            req = queues_[consumer_queue_index_].front();
            if (check_req_func(*req)) {
                queues_[consumer_queue_index_].pop();
            } else {
                req.reset();
            }
        }

        return req;
    }

private:
    int consumer_queue_index_;
    pthread_mutex_t switch_queue_lock_;
    std::queue<std::shared_ptr<ReqType>> queues_[2];
};

}}} // namespace ppl::llm::utils

#endif
