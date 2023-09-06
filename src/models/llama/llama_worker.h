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

#include "resource.h"
#include "common/server.h"
#include "utils/index_manager.h"
#include "utils/queue_request_scheduler.h"
#include "sampler/sampler.h"

#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#include "ppl/nn/runtime/tensor.h"
#include "ppl/common/threadpool.h"

#include <sentencepiece_processor.h>
#include <memory>
#include <iostream>
#include <unordered_map>

namespace ppl { namespace llm { namespace llama {

struct ModelConfig {
    int hidden_dim;
    int intermediate_dim;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int vocab_size;

    float norm_eps; // not used

    int cache_quant_bit;
    int cache_quant_group;

    int cache_layout;
    int cache_mode;

    bool dynamic_batching;
    bool auto_causal;
};

struct WorkerConfig {
    float top_p;
    int top_k;

    int max_tokens_per_request;
    int max_running_batch;
};

struct TidController {
    // init
    uint64_t tid;
    std::string prompt;

    float temperature;

    bool is_first_fill = true;
    int first_fill_len; // update
    int total_len;

    uint64_t cache_index;

    int rest_iters; // update
    int passed_iters = 0; // update

    std::vector<int> next_tokens; // update
};

struct WorkerController {
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
    std::vector<uint64_t> tid_finished; // iter

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
        tid_finished.clear();
    }
};

struct WorkerThreadArg {
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

struct LlamaRequest final {
    uint64_t uuid;
    Connection* conn;
    std::shared_ptr<Request> orig;
    std::vector<int> token_id_list;
};

class LLaMAWorker final : public RequestProcessor {
public:
    struct UuidData final {
        UuidData(uint64_t rid = UINT64_MAX, Connection* c = nullptr) : req_id(rid), conn(c) {}
        uint64_t req_id;
        Connection* conn;
    };

    LLaMAWorker(const sentencepiece::SentencePieceProcessor* tokenizer, const Resource& resource,
                const ModelConfig& mconfig, const WorkerConfig& wconfig);

    ~LLaMAWorker();

    ppl::common::RetCode Init();
    void ClearTask(Connection*);

    void Process(const std::shared_ptr<Request>&, Connection*) override;

    void SetSampler(const std::shared_ptr<Sampler>& sampler) {
        sampler_ = sampler;
    }

private:
    ppl::common::RetCode CheckParameters() const;
    void Work();
    void DeleteTask(const std::vector<uint64_t>& finished_list,
                    std::unordered_map<uint64_t, TidController>* tid_controllers);

private:
    static void* WorkerThreadFunc(void*);

private:
    const sentencepiece::SentencePieceProcessor* tokenizer_;

    ModelConfig model_config_;
    WorkerConfig worker_config_;

    // worker threads bound to specific devices
    ppl::common::ThreadPool* device_workers_;

    int tensor_parallel_size_;

    WorkerController worker_controller_;
    std::vector<WorkerThreadArg> worker_thread_args_;

    ppl::common::ThreadPool thread_pool_;
    pthread_mutex_t decoder_lock_;

    utils::IndexManager idx_mgr_;
    std::shared_ptr<Sampler> sampler_;

    uint64_t kv_cache_max_tokens_;

    bool worker_thread_created_ = false;
    pthread_t worker_thread_;

    pthread_cond_t req_signal_;

    utils::QueueRequestScheduler<LlamaRequest> sched_;
    pthread_mutex_t uuid_data_lock_;
    std::unordered_map<uint64_t, UuidData> uuid_data_;
    std::unordered_map<Connection*, std::vector<uint64_t>> conn2uuid_;
    uint64_t uuid_seq_ = 0;
};

}}} // namespace ppl::llm::llama

#endif
