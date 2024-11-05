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

#include "llm_engine.h"
#include "../common/profiler.h"
#include "../utils/utils.h"
#include "ppl/nn/engines/llm_cuda/options.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn;

namespace ppl { namespace llm {

RetCode SetInputTask(uint32_t id, uint32_t cache_mode, bool req_list_changed, const ModelInput& model_input,
                     EngineInferItem* infer_items) {
    infer_items[id].token_ids->FreeBuffer();
    infer_items[id].seq_starts->FreeBuffer();
    infer_items[id].kv_starts->FreeBuffer();
    infer_items[id].start_pos->FreeBuffer();
    infer_items[id].logits->FreeBuffer();

    RetCode rc;
    int32_t bs = model_input.start_pos.size();
    // token ids
    infer_items[id].token_ids->GetShape()->Reshape({int64_t(model_input.token_inputs.size())});
    rc = infer_items[id].token_ids->CopyFromHostAsync(model_input.token_inputs.data());
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set token_ids [" << infer_items[id].token_ids->GetName() << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }

    // seq_start
    infer_items[id].seq_starts->GetShape()->Reshape({int64_t(model_input.seq_starts.size())});
    rc = infer_items[id].seq_starts->CopyFromHostAsync(model_input.seq_starts.data());
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set seq_starts [" << infer_items[id].seq_starts->GetName() << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }

    // kv_starts
    infer_items[id].kv_starts->GetShape()->Reshape({int64_t(model_input.kv_starts.size())});
    rc = infer_items[id].kv_starts->CopyFromHostAsync(model_input.kv_starts.data());
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set kv_starts " << infer_items[id].kv_starts->GetName() << " failed: " << GetRetCodeStr(rc);
        return rc;
    }

    // cache_indices
    if (cache_mode == 0) {
        infer_items[id].cache_indices->GetShape()->Reshape({int64_t(model_input.cache_indices.size())});
        rc = infer_items[id].cache_indices->CopyFromHostAsync(model_input.cache_indices.data());
    } else if (cache_mode == 1) {
        if (req_list_changed) {
            infer_items[id].cache_indices->GetShape()->Reshape({int64_t(bs), int64_t(model_input.max_pages)});
            rc = infer_items[id].cache_indices->CopyFromHostAsync(model_input.page_list.data());
        }
    }
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set cache_indices [" << infer_items[id].cache_indices->GetName()
                   << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }

    // decoding batches
    rc = infer_items[id].decoding_batches->CopyFromHostAsync(&model_input.decoding_batches);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set decoding_batches [" << infer_items[id].decoding_batches->GetName()
                   << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }

    // start_pos
    infer_items[id].start_pos->GetShape()->Reshape({int64_t(model_input.start_pos.size())});
    rc = infer_items[id].start_pos->CopyFromHostAsync(model_input.start_pos.data());
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set start_pos [" << infer_items[id].start_pos->GetName() << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }

    // max_seq_len
    rc = infer_items[id].max_seq_len->CopyFromHostAsync(&model_input.max_seq_len);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set max_seq_len [" << infer_items[id].max_seq_len->GetName()
                   << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }

    // max_kv_len
    rc = infer_items[id].max_kv_len->CopyFromHostAsync(&model_input.max_kv_len);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set max_kv_len [" << infer_items[id].max_kv_len->GetName() << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }

    return rc;
}

static RetCode RunModelTask(uint32_t id, bool is_prefix_cache_hit, EngineInferItem* infer_items) {
    infer_items[id].nn_engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_CACHE_PREFILL, is_prefix_cache_hit ? 1 : 0);
    return infer_items[id].runtime->Run();
}

RetCode LLMEngine::Init(WorkerPerStepCounter* step_counter) {
    step_counter_ = step_counter;
    for (int i = 0; i < tensor_parallel_size_; i++) {
        auto item = &infer_items_[i];
        if (model_config_.cache_layout == 0) {
            item->kv_cache->GetShape()->Reshape({(int64_t)kv_cache_max_tokens_, model_config_.num_layers, 2,
                                                 model_config_.num_kv_heads / tensor_parallel_size_,
                                                 model_config_.hidden_dim / model_config_.num_heads});
            if (model_config_.cache_quant_bit > 0) {
                item->kv_scale->GetShape()->Reshape(
                    {(int64_t)kv_cache_max_tokens_, model_config_.num_layers, 2,
                     model_config_.num_kv_heads / tensor_parallel_size_,
                     model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});
            }

        } else if (model_config_.cache_layout == 1) {
            item->kv_cache->GetShape()->Reshape({model_config_.num_layers, (int64_t)kv_cache_max_tokens_, 2,
                                                 model_config_.num_kv_heads / tensor_parallel_size_,
                                                 model_config_.hidden_dim / model_config_.num_heads});
            if (model_config_.cache_quant_bit > 0) {
                item->kv_scale->GetShape()->Reshape(
                    {model_config_.num_layers, (int64_t)kv_cache_max_tokens_, 2,
                     model_config_.num_kv_heads / tensor_parallel_size_,
                     model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});
            }
        } else if (model_config_.cache_layout == 2) {
            item->kv_cache->GetShape()->Reshape({model_config_.num_layers, 2, (int64_t)kv_cache_max_tokens_,
                                                 model_config_.num_kv_heads / tensor_parallel_size_,
                                                 model_config_.hidden_dim / model_config_.num_heads});
            if (model_config_.cache_quant_bit > 0) {
                item->kv_scale->GetShape()->Reshape(
                    {model_config_.num_layers, 2, (int64_t)kv_cache_max_tokens_,
                     model_config_.num_kv_heads / tensor_parallel_size_,
                     model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});
            }
        } else if (model_config_.cache_layout == 3) {
            item->kv_cache->GetShape()->Reshape(
                {model_config_.num_layers, 2, model_config_.num_kv_heads / tensor_parallel_size_,
                 (int64_t)kv_cache_max_tokens_, model_config_.hidden_dim / model_config_.num_heads});
            if (model_config_.cache_quant_bit > 0) {
                item->kv_scale->GetShape()->Reshape(
                    {model_config_.num_layers, 2, model_config_.num_kv_heads / tensor_parallel_size_,
                     (int64_t)kv_cache_max_tokens_,
                     model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});
            }
        } else {
            LOG(ERROR) << "impossible status: cache_layout = [" << model_config_.cache_layout << "]";
            return RC_INVALID_VALUE;
        }
    }
    return RC_SUCCESS;
}

RetCode LLMEngine::Execute(const ModelInput& model_input, bool req_list_changed, bool is_prefix_cache_hit, ModelOutput* model_output,
                           std::string* error_msg) {
    int32_t running_batch = model_input.start_pos.size();
    // set inputs tensor
    RetCode rc;
    {
        utils::TimingGuard __timing__(&step_counter_->current.set_input_cost);

        rc = utils::ParallelExecute(SetInputTask, device_worker_pool_, model_config_.cache_mode, req_list_changed,
                                    model_input, infer_items_.data());
        if (rc != RC_SUCCESS) {
            *error_msg = "ParallelExecute(SetInputTask) failed: " + std::string(GetRetCodeStr(rc));
            LOG(ERROR) << *error_msg;
            return RC_OTHER_ERROR;
        }
    }
    step_counter_->global.set_input_cost += step_counter_->current.set_input_cost;
    // model forward
    {
        utils::TimingGuard __timing__(&step_counter_->current.model_forward_cost);
        rc = utils::ParallelExecute(RunModelTask, device_worker_pool_, is_prefix_cache_hit, infer_items_.data());
        if (rc != RC_SUCCESS) {
            *error_msg = "ParallelExecute(RunModelTask) failed: " + std::string(GetRetCodeStr(rc));
            LOG(ERROR) << *error_msg;
            return RC_OTHER_ERROR;
        }
    }
    step_counter_->global.model_forward_cost += step_counter_->current.model_forward_cost;

    auto* logits = infer_items_[0].logits;
    {
        utils::TimingGuard __timing__(&step_counter_->current.choose_token_cost);
        // penalty
        if (engine_config_.enable_penalty) {
            rc = post_processor_->ApplyPenalty(
                model_input.temperatures.data(), model_input.repetition_penalty_list.data(), nullptr, nullptr,
                model_input.batch_slots.data(), (int64_t*)infer_items_[0].token_ids->GetBufferPtr(),
                (int64_t*)infer_items_[0].seq_starts->GetBufferPtr(),
                (int64_t*)infer_items_[0].start_pos->GetBufferPtr(), running_batch, model_config_.vocab_size,
                req_list_changed, (float*)logits->GetBufferPtr());
            if (rc != RC_SUCCESS) {
                *error_msg = "Apply Penalty failed: " + std::string(GetRetCodeStr(rc));
                LOG(ERROR) << *error_msg;
                return RC_OTHER_ERROR;
            }
        }

        // sampling
        int32_t default_top_k = model_input.top_k_list.empty() ? engine_config_.top_k : model_input.top_k_list[0];
        rc = post_processor_->SampleTopKTopP(
            (float*)logits->GetBufferPtr(), model_input.temperatures.data(), model_input.top_k_list.data(),
            model_input.top_p_list.data(), running_batch, model_config_.vocab_size, logits->GetShape()->GetDim(1),
            default_top_k, engine_config_.top_p, req_list_changed, model_output->output_token.data(),
            model_output->logprobs.data(), engine_config_.enable_penalty);
        if (rc != RC_SUCCESS) {
            *error_msg = "SampleTopKTopP failed: " + std::string(GetRetCodeStr(rc));
            LOG(ERROR) << *error_msg;
            return RC_OTHER_ERROR;
        }
    }
    step_counter_->global.choose_token_cost += step_counter_->current.choose_token_cost;
    step_counter_->current.output_token_cnt = running_batch;
    step_counter_->global.output_token_cnt += step_counter_->current.output_token_cnt;

    return RC_SUCCESS;
}

}} // namespace ppl::llm