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

#ifndef __PPL_LLM_LLM_GENERATOR_H__
#define __PPL_LLM_LLM_GENERATOR_H__

#include "../engine/llm_engine.h"
#include "../common/config.h"
#include "../common/request.h"
#include "../common/connection.h"
#include "../common/resource.h"
#include "../common/post_processor.h"
#include "../tokenizer/tokenizer.h"
#include "../utils/index_manager.h"
#include "../utils/mpsc_request_scheduler.h"
#include "../utils/prefix_cache_manager.h"
#include "../common/profiler.h"

#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#include "ppl/nn/runtime/tensor.h"
#include "ppl/common/threadpool.h"
#include "ppl/common/typed_mpsc_queue.h"
#include "ppl/common/event_count.h"
#include "ppl/common/page_manager.h"

#include <chrono>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <atomic>

namespace ppl { namespace llm {

struct TidGenToken final {
    TidGenToken(uint64_t _tid, int _token, float _logprob, FinishFlag _finish_flag, uint64_t _steps,
                bool _is_token_in_out, bool _is_special)
        : tid(_tid)
        , token(_token)
        , logprob(_logprob)
        , finish_flag(_finish_flag)
        , steps(_steps)
        , is_token_in_out(_is_token_in_out)
        , is_special(_is_special) {}
    uint64_t tid;
    int token;
    float logprob;
    FinishFlag finish_flag;
    uint64_t steps;
    bool is_token_in_out;
    bool is_special;
};

struct FinishedTaskInfo final {
    FinishedTaskInfo(uint64_t fid = UINT64_MAX, uint32_t ftype = UNKNOWN) : id(fid), type(ftype) {}
    uint64_t id;
    enum {
        UNKNOWN,
        FROM_WORKER,
        FROM_CONN,
    };
    uint32_t type;
};

struct TidData final {
    uint64_t tid;
    float temperature;
    float top_p;
    int32_t top_k;
    float repetition_penalty;
    float presence_penalty;
    float frequency_penalty;
    bool early_stopping;
    int32_t rest_iters;
    bool is_token_in_out = false;
    int64_t total_len;
    std::shared_ptr<std::unordered_set<int>> stop_tokens;

    std::shared_ptr<std::vector<int>> next_tokens;
    // int64_t seqlen;
    int64_t start_pos;
    uint64_t cache_index;
    std::vector<int64_t> page_list;
    int64_t slot_index;
    int32_t steps;
    int32_t gen_tokens_cnt = 0;
    std::vector<uint64_t> hash_list;
    int64_t cache_hit_count = 0;
};

struct Controller {
    bool req_list_changed = true;
    ppl::common::TypedMPSCQueue<FinishedTaskInfo> finished_tasks;
    std::vector<TidData*> tid_list;
    void Reset() {
        tid_list.clear();
        req_list_changed = true;
        while (true) {
            FinishedTaskInfo info;
            bool ok = finished_tasks.Pop(&info);
            if (!ok) {
                break;
            }
        }
    }
};

struct LlmRequest final : public ppl::common::MPSCQueue::Node {
    std::shared_ptr<Request> orig;
    std::chrono::time_point<std::chrono::high_resolution_clock> enqueue_ts;
};

class LLMGenerator final {
public:
    LLMGenerator(const Resource& resource, const GeneratorConfig& generator_config, const ModelConfig& model_config,
                 Connection* conn);

    ~LLMGenerator() {
        bool is_active = generate_thread_active_.load(std::memory_order_relaxed);
        if (is_active) {
            generate_thread_active_.store(false, std::memory_order_release);
            req_signal_.NotifyOne();
            pthread_join(generate_thread_, nullptr);
        }
    }
    ppl::common::RetCode Init();
    void Process(const std::shared_ptr<Request>&);

    void ClearTask(uint64_t tid) {
        controller_.finished_tasks.Push(FinishedTaskInfo(tid, FinishedTaskInfo::FROM_CONN));
    }

    uint32_t GetPendingTaskNum() const {
        return sched_.GetPendingSize();
    }

private:
    ppl::common::RetCode CheckParameters() const;
    void Generate();
    void DeleteTasks(ModelInput*, ppl::common::TypedMPSCQueue<FinishedTaskInfo>*, std::map<uint64_t, TidData>*,
                     uint64_t*, uint64_t*);
    void ReleaseResource();

private:
    static void* GeneratorThreadFunc(void*);

private:
    const Tokenizer* tokenizer_;
    GeneratorConfig generator_config_;
    ModelConfig model_config_;
    Connection* conn_;
    uint64_t kv_cache_max_tokens_ = 0;
    LLMEngine llm_engine_;
    ppl::common::StaticThreadPool decoder_thread_pool_;

    Controller controller_;
    utils::IndexManager idx_mgr_;
    utils::IndexManager batch_slots_mgr_;
    ppl::common::PageManager page_mgr_;
    std::shared_ptr<WorkerProfiler> worker_profiler_;

    std::atomic<bool> generate_thread_active_ = {false};
    pthread_t generate_thread_;
    ppl::common::EventCount req_signal_;
    utils::MPSCRequestScheduler<LlmRequest> sched_;
    utils::PrefixCacheManager prefix_cache_mgr_;
    static constexpr int DECODER_THREAD_NUM = 1;
};

}} // namespace ppl::llm

#endif