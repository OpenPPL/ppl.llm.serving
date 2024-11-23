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

#ifndef __PPL_LLM_LLM_ENGINE_H__
#define __PPL_LLM_LLM_ENGINE_H__

#include "../common/resource.h"
#include "../common/config.h"
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

struct ModelInput {
    int64_t decoding_batches = 0;
    int64_t max_seq_len = 0;
    int64_t max_kv_len = 0;
    int64_t max_pages = 0;

    std::vector<int64_t> token_inputs;
    std::vector<int64_t> seq_starts;
    std::vector<int64_t> start_pos;
    std::vector<int64_t> cache_indices;
    std::vector<int64_t> page_list;
    std::vector<int64_t> kv_starts;
    std::vector<float> temperatures;
    std::vector<float> top_p_list;
    std::vector<int32_t> top_k_list;

    std::vector<float> repetition_penalty_list;
    std::vector<float> presence_penalty_list;
    std::vector<float> frequency_penalty_list;
    std::vector<int64_t> batch_slots;
};

struct ModelOutput {
    std::vector<int32_t> output_token;
    std::vector<float> logprobs;
    void Clear() {
        output_token.clear();
        logprobs.clear();
    }
    void Resize(int32_t n) {
        output_token.resize(n);
        logprobs.resize(n);
    }
};

struct EngineInferItem {
    void* kv_cache_mem = nullptr;
    void* kv_scale_mem = nullptr;
    ppl::nn::Runtime* runtime = nullptr;
    ppl::nn::DeviceContext* host_device = nullptr;
    ppl::nn::Engine* nn_engine = nullptr;

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

struct EngineConfig {
    EngineConfig(bool _enable_penalty, int32_t _top_k, float _top_p)
        : enable_penalty(_enable_penalty), top_k(_top_k), top_p(_top_p) {}

    bool enable_penalty;
    int32_t top_k;
    float top_p;
};
class LLMEngine final {
public:
    LLMEngine(const Resource& resource, const ModelConfig& model_config, bool enable_penalty, int32_t top_k,
              float top_p)
        : tensor_parallel_size_(resource.tensor_parallel_size)
        , device_worker_pool_(resource.device_worker_pool_)
        , infer_items_(resource.tensor_parallel_size)
        , kv_cache_max_tokens_(resource.kv_cache_max_tokens)
        , post_processor_(resource.post_processor)
        , model_config_(model_config)
        , engine_config_(enable_penalty, top_k, top_p) {
        for (int i = 0; i < tensor_parallel_size_; i++) {
            auto* item = &infer_items_[i];
            item->kv_cache_mem = resource.items[i].kv_cache_mem;
            item->kv_scale_mem = resource.items[i].kv_scale_mem;
            item->runtime = resource.items[i].runtime;
            item->host_device = resource.items[i].host_device;
            item->nn_engine = resource.items[i].engine;

            item->token_ids = item->runtime->GetInputTensor(0);
            item->attn_mask = item->runtime->GetInputTensor(1);
            item->seq_starts = item->runtime->GetInputTensor(2);
            item->kv_starts = item->runtime->GetInputTensor(3);
            item->cache_indices = item->runtime->GetInputTensor(4);
            item->decoding_batches = item->runtime->GetInputTensor(5);
            item->start_pos = item->runtime->GetInputTensor(6);
            item->max_seq_len = item->runtime->GetInputTensor(7);
            item->max_kv_len = item->runtime->GetInputTensor(8);
            item->kv_cache = item->runtime->GetInputTensor(9);
            if (model_config_.cache_quant_bit > 0) {
                item->kv_scale = item->runtime->GetInputTensor(10);
            }

            item->logits = item->runtime->GetOutputTensor(0);

            item->decoding_batches->SetDeviceContext(item->host_device);
            item->max_seq_len->SetDeviceContext(item->host_device);
            item->max_kv_len->SetDeviceContext(item->host_device);

            item->kv_cache->SetBufferPtr(item->kv_cache_mem);
            if (model_config_.cache_quant_bit > 0) {
                item->kv_scale->SetBufferPtr(item->kv_scale_mem);
            }
        }
    }

public:
    ppl::common::RetCode Init(WorkerPerStepCounter*);
    ppl::common::RetCode Execute(const ModelInput&, bool, bool, ModelOutput*, std::string*);

private:
    int32_t tensor_parallel_size_;
    ppl::common::StaticThreadPool* device_worker_pool_;
    std::vector<EngineInferItem> infer_items_;
    uint64_t kv_cache_max_tokens_;
    PostProcessor* post_processor_;
    ModelConfig model_config_;
    EngineConfig engine_config_;
    WorkerPerStepCounter* step_counter_;
};

}} // namespace ppl::llm

#endif