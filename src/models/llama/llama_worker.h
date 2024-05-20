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

#ifndef __PPL_LLM_LLAMA_WORKER_H__
#define __PPL_LLM_LLAMA_WORKER_H__

#include "../../common/processor.h"
#include "../../models/config.h"
#include "../../models/resource.h"
#include "../../utils/index_manager.h"
#include "../../utils/mpsc_request_scheduler.h"
#include "../../utils/sampler.h"
#include "../../utils/tokenizer.h"

#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#include "ppl/nn/runtime/tensor.h"
#include "ppl/common/threadpool.h"
#include "ppl/common/typed_mpsc_queue.h"

#include <memory>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace ppl { namespace llm { namespace llama {

struct TidGenToken final {
    TidGenToken(uint64_t _tid, int _token, bool _is_last, bool _is_token_in_out) : tid(_tid), token(_token), is_last(_is_last), is_token_in_out(_is_token_in_out) {}
    uint64_t tid;
    int token;
    bool is_last;
    bool is_token_in_out;
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

struct TidController final {
    // init
    uint64_t tid;
    float temperature;
    bool early_stopping;

    bool is_first_fill = true;
    int first_fill_len; // update
    int total_len;

    uint64_t cache_index;

    int rest_iters; // update
    int passed_iters = 0; // update

    std::shared_ptr<std::vector<int>> next_tokens; // update
    std::shared_ptr<std::unordered_set<int>> stop_tokens;
    bool is_token_in_out = false;
};

struct WorkerController final {
    int64_t decoding_batches = 0; // update && finish
    int64_t max_seq_len = 0; // iter
    int64_t max_kv_len = 0; // iter

    int total_rest_iters = 0; // iter && incoming

    std::vector<TidController*> tid_list;

    std::vector<int64_t> token_inputs; // iter
    std::vector<int64_t> seq_starts; // iter
    std::vector<int64_t> start_pos; // update && incoming && finish
    std::vector<int64_t> cache_indices; // incoming && finish
    std::vector<int64_t> kv_starts; // iter
    std::vector<float> temperatures; // iter
    ppl::common::TypedMPSCQueue<FinishedTaskInfo> finished_tasks;

    void Reset() {
        decoding_batches = 0;
        max_seq_len = 0;
        max_kv_len = 0;

        total_rest_iters = 0;

        tid_list.clear();

        token_inputs.clear();
        seq_starts.clear();
        start_pos.clear();
        cache_indices.clear();
        kv_starts.clear();

        temperatures.clear();
    }
};

struct WorkerThreadArg final {
    ResourceItem* resource;

    ppl::nn::Tensor* token_ids;
    ppl::nn::Tensor* attn_mask;
    ppl::nn::Tensor* seq_starts;
    ppl::nn::Tensor* kv_starts;
    ppl::nn::Tensor* cache_indices;
    ppl::nn::Tensor* decoding_batches;
    ppl::nn::Tensor* start_pos;
    ppl::nn::Tensor* max_seq_len;
    ppl::nn::Tensor* max_kv_len;
    ppl::nn::Tensor* kv_cache;
    ppl::nn::Tensor* kv_scale;

    ppl::nn::Tensor* logits;
};

struct LlamaRequest final : public ppl::common::MPSCQueue::Node {
    std::shared_ptr<Request> orig;
    std::shared_ptr<std::vector<int>> token_id_list;
    std::shared_ptr<std::unordered_set<int>> stop_tokens;
    bool is_token_in_out;
};

class LLaMAWorker final : public RequestProcessor {
public:
    LLaMAWorker(const Resource& resource, const ModelConfig& mconfig, const WorkerConfig& wconfig,
                Connection*);

    ~LLaMAWorker();

    ppl::common::RetCode Init();
    void ClearTask(uint64_t) override;

    void Process(const std::shared_ptr<Request>&) override;

private:
    ppl::common::RetCode CheckParameters() const;
    void Work();
    void DeleteTasks(ppl::common::TypedMPSCQueue<FinishedTaskInfo>* finished_tasks,
                     std::unordered_map<uint64_t, TidController>* tid_controllers,
                     int* accu_task_count);

private:
    static void* WorkerThreadFunc(void*);

private:
    const utils::Tokenizer* tokenizer_;

    ModelConfig model_config_;
    WorkerConfig worker_config_;

    // worker threads bound to specific devices
    ppl::common::StaticThreadPool* device_worker_pool_;

    int tensor_parallel_size_;

    WorkerController worker_controller_;
    std::vector<WorkerThreadArg> worker_thread_args_;

    ppl::common::StaticThreadPool decoder_thread_pool_;

    utils::IndexManager idx_mgr_;
    utils::Sampler* sampler_;

    uint64_t kv_cache_max_tokens_;

    std::atomic<bool> worker_thread_active_ = {false};
    pthread_t worker_thread_;

    ppl::common::EventCount req_signal_;

    utils::MPSCRequestScheduler<LlamaRequest> sched_;

private:
    static constexpr int DECODER_THREAD_NUM = 2;

private:
    LLaMAWorker(const LLaMAWorker&) = delete;
    void operator=(const LLaMAWorker&) = delete;
    LLaMAWorker(LLaMAWorker&&) = delete;
    void operator=(LLaMAWorker&&) = delete;
};

}}} // namespace ppl::llm::llama

#endif
