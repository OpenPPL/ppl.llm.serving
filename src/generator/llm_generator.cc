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

#include "llm_generator.h"
#include "../utils/utils.h"
#include <unistd.h>
#include <cmath>
#include <limits>
#ifdef PPLNN_USE_LLM_CUDA
#include <cuda_runtime.h>
#endif
using namespace std;
using namespace ppl::common;
using namespace ppl::nn;

namespace ppl { namespace llm {

struct RequestCheckResult final {
    int64_t cache_index;
    std::vector<int64_t> page_list;
    int64_t slot_index;
    int rest_iters;
    int first_fill_len;
    int total_tokens_per_step;
    std::vector<uint64_t> hash_list;
    int64_t cache_hit_count;
    int32_t running_batch;
    int32_t prefill_batch;
    string errmsg;
};

#ifdef PPL_LLM_ENABLE_DEBUG
#include <unistd.h>
template <class T>
static void PrintVector(vector<T> vec, const std::string& prefix = "") {
    stringstream ss;
    for (auto& ele : vec) {
        ss << ele << ", ";
    }
    std::cout << prefix << ": " << ss.str() << std::endl;
}
#endif

static RetCode DecodeAndSendTask(uint32_t start_id, uint32_t end_id, const Tokenizer* tokenizer,
                                 const std::vector<TidGenToken>& tid_gen_token_list, Connection* conn , map<int,int>* decode_stat, map<int,vector<int>>* decode_buffer) {
    // for extreme case task_num < thread_num
    if (start_id >= tid_gen_token_list.size()) {
        return RC_SUCCESS;
    }

    vector<Response> rsp_list(end_id - start_id);
    for (uint32_t i = start_id; i < end_id; ++i) {
        auto it = decode_stat->find(tid_gen_token_list[i].tid);
        if (it == decode_stat->end()) {
            decode_stat->insert(pair<int, int>(tid_gen_token_list[i].tid, 0));
            decode_buffer->insert(pair<int, vector<int>>(tid_gen_token_list[i].tid, {0,0,0}));
        }
    }

    for (uint32_t i = start_id; i < end_id; ++i) {
        const auto& tid_gen_token = tid_gen_token_list[i];
        int token = tid_gen_token.token;

        Response& rsp = rsp_list[i - start_id];
        // Response rsp;
        rsp.token = token;
        if (!tid_gen_token.is_token_in_out) {
            tokenizer->Decode(&token, 1, &rsp.generated);
        }
        const char *uft8_unkown = "\xef\xbf\xbd";
        int& flag = (*decode_stat)[tid_gen_token_list[i].tid];
        auto& buffer = (*decode_buffer)[tid_gen_token_list[i].tid];

        if (strcmp(uft8_unkown, rsp.generated.c_str()) == 0) {
            if (flag < 3) {
                buffer[flag] = token;
                flag++;
                rsp.generated = "";
                if (flag == 3) {
                    tokenizer->Decode(buffer.data(), 3, &rsp.generated);
                    flag = 0;
                    buffer.assign(3, 0);
                }
            }
        }

        rsp.id = tid_gen_token.tid;
        rsp.finish_flag = tid_gen_token.finish_flag;
        rsp.logprob = tid_gen_token.logprob;
        rsp.is_special = tid_gen_token.is_special;
        if (rsp.finish_flag != ppl::llm::FinishFlag::NOT_FINISHED) {
            decode_stat->erase(tid_gen_token.tid);
            decode_buffer->erase(tid_gen_token.tid);
        }
    }
    conn->Send(rsp_list);
    return RC_SUCCESS;
}

RetCode LLMGenerator::CheckParameters() const {
    if (model_config_.auto_causal != true) {
        LOG(ERROR) << "only support auto_causal == true";
        return RC_INVALID_VALUE;
    }

    if (model_config_.cache_mode != 0 && model_config_.cache_mode != 1) {
        LOG(ERROR) << "unsupported cache_mode: " << model_config_.cache_mode;
        return RC_INVALID_VALUE;
    }

    if (model_config_.cache_layout != 0 && model_config_.cache_layout != 1 && model_config_.cache_layout != 2 &&
        model_config_.cache_layout != 3) {
        LOG(ERROR) << "only support cache_layout == 0 || cache_layout == 1 || cache_layout == 2 || cache_layout == 3";
        return RC_INVALID_VALUE;
    }

    if ((model_config_.cache_quant_bit != 8 || model_config_.cache_quant_group != 8) &&
        (model_config_.cache_quant_bit != 0 || model_config_.cache_quant_group != 1)) {
        LOG(ERROR) << "only support (cache_quant_bit == 8 and cache_quant_group == 8) or (cache_quant_bit == 1 and "
                      "cache_quant_group == 0)";
        return RC_INVALID_VALUE;
    }

    if (model_config_.dynamic_batching != true) {
        LOG(ERROR) << "only support dynamic_batching == true";
        return RC_INVALID_VALUE;
    }

    return RC_SUCCESS;
}

LLMGenerator::LLMGenerator(const Resource& resource, const GeneratorConfig& generator_config,
                           const ModelConfig& model_config, Connection* conn)
    : tokenizer_(resource.tokenizer)
    , generator_config_(generator_config)
    , model_config_(model_config)
    , conn_(conn)
    , kv_cache_max_tokens_(resource.kv_cache_max_tokens)
    , llm_engine_(resource, model_config, generator_config.enable_penalty, generator_config_.top_k,
                  generator_config_.top_p) {
    idx_mgr_.Init(kv_cache_max_tokens_);
    batch_slots_mgr_.Init(generator_config.max_running_batch);
    page_mgr_.Init(kv_cache_max_tokens_, model_config_.page_size);

    worker_profiler_ = std::make_shared<WorkerProfiler>();
}

RetCode LLMGenerator::Init() {
    auto ret = CheckParameters();
    if (ret != RC_SUCCESS) {
        LOG(ERROR) << "CheckParameters failed.";
        return ret;
    }

    ret = llm_engine_.Init(&worker_profiler_->step_counter);
    if (ret != RC_SUCCESS) {
        LOG(ERROR) << "LLM Engine Init failed.";
        return ret;
    }

#ifndef PPL_LLM_SERVING_SYNC_DECODE
    ret = decoder_thread_pool_.Init(DECODER_THREAD_NUM);
    if (ret != RC_SUCCESS) {
        LOG(ERROR) << "Init decoder thread pool error";
        return RC_OTHER_ERROR;
    }
#endif

    generate_thread_active_.store(true, std::memory_order_release);
    auto err = pthread_create(&generate_thread_, nullptr, GeneratorThreadFunc, this);
    if (err != 0) {
        generate_thread_active_.store(false, std::memory_order_relaxed);
        LOG(ERROR) << "create generator thread failed.";
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

static bool ParseRequest(const LlmRequest& req, const RequestCheckResult& check_res, int cache_mode,
                         ModelInput* model_input, Controller* controller, map<uint64_t, TidData>* tid_data_map,
                         Connection* conn) {
    if (check_res.rest_iters <= 0 || check_res.first_fill_len == -1) {
        conn->NotifyFailure(req.orig->id, RC_INVALID_VALUE, check_res.errmsg);
        return true;
    }

    if ((cache_mode == 0 && check_res.cache_index == INT64_MAX) ||
        (cache_mode == 1 && check_res.page_list.size() == 0)) {
        LOG(ERROR) << "catch invalid cache_index or page list";
        return false;
    }

    auto tid = req.orig->id;
    auto& tid_data = tid_data_map->emplace(tid, TidData()).first->second;
    tid_data.tid = tid;
    tid_data.temperature = req.orig->temperature;
    tid_data.top_p = req.orig->top_p;
    tid_data.top_k = req.orig->top_k;
    tid_data.early_stopping = req.orig->early_stopping;
    tid_data.rest_iters = check_res.rest_iters;
    tid_data.total_len = check_res.first_fill_len + check_res.rest_iters;
    tid_data.stop_tokens = req.orig->stop_tokens;
    tid_data.is_token_in_out = req.orig->is_token_in_out;
    tid_data.repetition_penalty = req.orig->repetition_penalty;
    tid_data.presence_penalty = req.orig->presence_penalty;
    tid_data.frequency_penalty = req.orig->frequency_penalty;
    if (cache_mode == 0) {
        tid_data.cache_index = check_res.cache_index;
    } else {
        tid_data.page_list = check_res.page_list;
        tid_data.hash_list = check_res.hash_list;
        tid_data.cache_hit_count = check_res.cache_hit_count;
    }

    if (check_res.cache_hit_count == 0) {
        tid_data.next_tokens = req.orig->token_ids;
        tid_data.start_pos = 0;
        model_input->start_pos.push_back(0);
    } else if (size_t(check_res.cache_hit_count) == req.orig->token_ids->size()) {
        tid_data.next_tokens = std::shared_ptr<vector<int>>(new vector<int>({req.orig->token_ids->back()}));
        model_input->start_pos.push_back(0 + check_res.cache_hit_count - 1);
        tid_data.start_pos = check_res.cache_hit_count - 1;
    } else {
        tid_data.next_tokens = std::make_shared<std::vector<int>>(
            req.orig->token_ids->begin() + check_res.cache_hit_count, req.orig->token_ids->end());
        model_input->start_pos.push_back(0 + check_res.cache_hit_count);
        tid_data.start_pos = check_res.cache_hit_count;
    }

    tid_data.slot_index = check_res.slot_index;

    controller->tid_list.push_back(&tid_data);
    model_input->temperatures.push_back(tid_data.temperature);
    model_input->top_p_list.push_back(tid_data.top_p);
    model_input->top_k_list.push_back(tid_data.top_k);
    model_input->repetition_penalty_list.push_back(tid_data.repetition_penalty);
    model_input->presence_penalty_list.push_back(tid_data.presence_penalty);
    model_input->frequency_penalty_list.push_back(tid_data.frequency_penalty);
    if (cache_mode == 0) {
        model_input->cache_indices.push_back(check_res.cache_index);
    } else {
        model_input->max_pages = std::max<int64_t>(tid_data.page_list.size(), model_input->max_pages);
    }

    model_input->batch_slots.push_back(tid_data.slot_index);
    return true;
}

static void UpdateInput(const Controller& controller, int cache_mode, bool req_list_changed, ModelInput* model_input) {
    // update input
    int running_batch = controller.tid_list.size();

    model_input->max_seq_len = 0;
    model_input->max_kv_len = 0;
    model_input->token_inputs.clear();
    model_input->seq_starts.clear();
    model_input->seq_starts.reserve(running_batch + 1);
    model_input->seq_starts.push_back(0);

    model_input->kv_starts.clear();
    model_input->kv_starts.reserve(running_batch + 1);
    model_input->kv_starts.push_back(0);

    if (req_list_changed && cache_mode == 1) {
        model_input->page_list.clear();
        model_input->page_list.resize(running_batch * model_input->max_pages, INT64_MAX);
    }

    for (int i = 0; i < running_batch; ++i) {
        const auto* tid_data = controller.tid_list[i];
        int32_t seqlen = tid_data->next_tokens->size();
        model_input->token_inputs.insert(model_input->token_inputs.end(), tid_data->next_tokens->begin(),
                                         tid_data->next_tokens->end());
        model_input->seq_starts.push_back(model_input->seq_starts[i] + seqlen);
        model_input->kv_starts.push_back(model_input->kv_starts[i] + tid_data->start_pos + seqlen);
        model_input->max_seq_len = std::max<int64_t>(tid_data->next_tokens->size(), model_input->max_seq_len);
        model_input->max_kv_len = std::max<int64_t>(tid_data->start_pos + seqlen, model_input->max_kv_len);

        if (req_list_changed && cache_mode == 1) {
            memcpy(model_input->page_list.data() + i * model_input->max_pages,
                   tid_data->page_list.data(), tid_data->page_list.size() * sizeof(int64_t));
        }
    }
}

static int RemoveFinishedTask(int cache_mode, ModelInput* model_input, Controller* controller) {
    int running_batch = controller->tid_list.size();
    int left = 0;
    if (cache_mode == 1) {
        model_input->max_pages = 0;
    }
    for (int right = 0; right < running_batch; right++) {
        if (controller->tid_list[right] != nullptr) {
            controller->tid_list[left] = controller->tid_list[right];
            if (cache_mode == 0) {
                model_input->cache_indices[left] = model_input->cache_indices[right];
            } else {
                model_input->max_pages =
                    std::max<int64_t>(model_input->max_pages, controller->tid_list[right]->page_list.size());
            }
            model_input->start_pos[left] = model_input->start_pos[right];
            model_input->temperatures[left] = model_input->temperatures[right];
            model_input->top_p_list[left] = model_input->top_p_list[right];
            model_input->top_k_list[left] = model_input->top_k_list[right];
            model_input->repetition_penalty_list[left] = model_input->repetition_penalty_list[right];
            model_input->presence_penalty_list[left] = model_input->presence_penalty_list[right];
            model_input->frequency_penalty_list[left] = model_input->frequency_penalty_list[right];
            model_input->batch_slots[left] = model_input->batch_slots[right];
            ++left;
        }
    }
    controller->tid_list.resize(left);
    if (cache_mode == 0) {
        model_input->cache_indices.resize(left);
    }
    model_input->start_pos.resize(left);
    model_input->temperatures.resize(left);
    model_input->top_p_list.resize(left);
    model_input->top_k_list.resize(left);
    model_input->repetition_penalty_list.resize(left);
    model_input->presence_penalty_list.resize(left);
    model_input->frequency_penalty_list.resize(left);
    model_input->batch_slots.resize(left);
    LOG(DEBUG) << "Rest tasks: " << controller->tid_list.size();
    return left;
}

void* LLMGenerator::GeneratorThreadFunc(void* arg) {
    auto generator = (LLMGenerator*)arg;
    while (true) {
        while (true) {
            auto wait_key = generator->req_signal_.PrepareWait();
            bool is_active = generator->generate_thread_active_.load(std::memory_order_acquire);
            if (!is_active) {
                generator->req_signal_.CancelWait();
                return nullptr;
            }

            if (generator->sched_.GetPendingSize() > 0) {
                generator->req_signal_.CancelWait();
                break;
            }

            LOG(INFO) << "waiting for request ...";
            generator->req_signal_.CommitWait(wait_key);
        }

        generator->Generate();
    }

    return nullptr;
}

void LLMGenerator::ReleaseResource() {
    for (size_t task_iter = 0; task_iter < controller_.tid_list.size(); ++task_iter) {
        if (model_config_.cache_mode == 0) {
            uint64_t cache_index = controller_.tid_list[task_iter]->cache_index;
            uint64_t total_len = controller_.tid_list[task_iter]->total_len;
            idx_mgr_.Free(cache_index, total_len - 1);
        } else {
            const auto& page_list = controller_.tid_list[task_iter]->page_list;
            page_mgr_.Free(page_list.data(), page_list.size());
        }
        if (generator_config_.enable_penalty) {
            uint64_t slots_index = controller_.tid_list[task_iter]->slot_index;
            batch_slots_mgr_.Free(slots_index, 1);
        }
    }
    prefix_cache_mgr_.Reset();
    controller_.Reset();
}

void LLMGenerator::DeleteTasks(ModelInput* model_input, TypedMPSCQueue<FinishedTaskInfo>* finished_tasks,
                               map<uint64_t, TidData>* tid_controllers, uint64_t* finished_task_cnt,
                               uint64_t* output_tokens_per_req_counter) {
    while (true) {
        FinishedTaskInfo info;
        bool ok = finished_tasks->Pop(&info);
        if (!ok) {
            break;
        }

        uint64_t tid = info.id;
        auto tid_it = tid_controllers->find(tid);
        if (tid_it == tid_controllers->end()) { // corner case: tid in finished_list and shutdown_list at same time
            continue;
        }

        auto& tid_ctrl = tid_it->second;
        --model_input->decoding_batches;

        // search deleted element
        size_t task_iter = 0;
        for (; task_iter < controller_.tid_list.size(); ++task_iter) {
            if (controller_.tid_list[task_iter] != nullptr && controller_.tid_list[task_iter]->tid == tid) {
                break;
            }
        }
        if (task_iter == controller_.tid_list.size()) {
            continue;
        }
        controller_.tid_list[task_iter] = nullptr;
        if (model_config_.cache_mode == 0) {
            idx_mgr_.Free(tid_ctrl.cache_index, tid_ctrl.total_len - 1);
        } else {
            if (generator_config_.enable_prefix_cache) {
                int64_t num_prefix_pages = tid_ctrl.hash_list.size();
                prefix_cache_mgr_.DecRefCount(tid_ctrl.hash_list.data(), num_prefix_pages);
                page_mgr_.Free(tid_ctrl.page_list.data() + num_prefix_pages, tid_ctrl.page_list.size() - num_prefix_pages);
            } else {
                page_mgr_.Free(tid_ctrl.page_list.data(), tid_ctrl.page_list.size());
            }
        }
        if (generator_config_.enable_penalty) {
            int64_t index = model_input->batch_slots[task_iter];
            batch_slots_mgr_.Free(index, 1);
        }

        output_tokens_per_req_counter += tid_it->second.gen_tokens_cnt;

        tid_controllers->erase(tid_it);
        ++(*finished_task_cnt);
    }
    RemoveFinishedTask(model_config_.cache_mode, model_input, &controller_);
}

static bool CheckTotalLen(const GeneratorConfig& generator_config, const LlmRequest& req,
                          RequestCheckResult* check_res) {
    if (check_res->first_fill_len > generator_config.max_input_tokens_per_request) {
        check_res->errmsg = "id [" + std::to_string(req.orig->id) +
            "] invalid input token len: " + std::to_string(check_res->first_fill_len) +
            ", server allowed max input len: " + std::to_string(generator_config.max_input_tokens_per_request);
        check_res->first_fill_len = -1;
        return false;
    }

    check_res->rest_iters = req.orig->generation_length;
    if (req.orig->generation_length > generator_config.max_output_tokens_per_request) {
        const std::string warning_msg = "id [" + std::to_string(req.orig->id) + "]: generation len in request is [" +
            std::to_string(req.orig->generation_length) + "] > [" +
            std::to_string(generator_config.max_output_tokens_per_request) + "] from cmd. use [" +
            std::to_string(generator_config.max_output_tokens_per_request) + "]";
        LOG(WARNING) << warning_msg;
        check_res->rest_iters = generator_config.max_output_tokens_per_request;
        if (check_res->rest_iters <= 0) {
            check_res->errmsg = warning_msg;
            return false;
        }
    }

    if (check_res->first_fill_len + req.orig->generation_length > generator_config.max_total_tokens_per_request) {
        const std::string warning_msg = "id [" + std::to_string(req.orig->id) + "]: total len in request is [" +
            std::to_string(check_res->first_fill_len + req.orig->generation_length) + "] > [" +
            std::to_string(generator_config.max_total_tokens_per_request) + "] from cmd. use [" +
            std::to_string(generator_config.max_total_tokens_per_request - check_res->first_fill_len) + "]";
        LOG(WARNING) << warning_msg;
        check_res->rest_iters = generator_config.max_total_tokens_per_request - check_res->first_fill_len;
        if (check_res->rest_iters <= 0) {
            check_res->errmsg = warning_msg;
            return false;
        }
    }
    return true;
}

static bool CheckAndAllocGPUMemory(const ModelConfig& model_config, const GeneratorConfig& generator_config,
                           const LlmRequest& req, int32_t running_batch, utils::IndexManager* idx_mgr,
                           PageManager* page_mgr, utils::PrefixCacheManager* prefix_cache_mgr, utils::IndexManager* batch_slots_mgr, int32_t* cache_cool_down_count,
                           RequestCheckResult* check_res, WorkerProfiler* worker_profiler, bool* is_prefix_cache_hit) {
    uint64_t total_len = check_res->first_fill_len + check_res->rest_iters - 1;
    LOG(DEBUG) << "total_len: " << total_len;
    if (model_config.cache_mode == 0) {
        check_res->cache_index = idx_mgr->Alloc(total_len);
        if (check_res->cache_index == INT64_MAX) {
            *cache_cool_down_count =
                std::min(std::max(1, (int)floorf(running_batch * 0.1f)), generator_config.max_cooldown_request);
            return false;
        }
        LOG(INFO) << "task[" << req.orig->id << "] cache index: " << check_res->cache_index << ", "
                  << "total_len: " << total_len;
    } else {
        if (generator_config.enable_prefix_cache) {
            auto token_ids = req.orig->token_ids;
            int64_t page_size = model_config.page_size;
            uint64_t prev_hash = 0;
            uint64_t start = 0;
            for (; start < token_ids->size(); start+=page_size) {
                if (start + page_size > token_ids->size()) {
                    break;
                }
                uint64_t hash_val = utils::HashCombine(prev_hash, token_ids->data() + start, page_size);
                int64_t page_id = prefix_cache_mgr->Find(hash_val);
                if (page_id == -1) {
                    break;
                }
                prev_hash = hash_val;
                check_res->page_list.push_back(page_id);
                check_res->hash_list.push_back(hash_val);
            }
            prefix_cache_mgr->IncRefCount(check_res->hash_list.data(), check_res->hash_list.size());

            int64_t avail_pages = page_mgr->GetAvail();
            int64_t need_pages = (total_len - start + page_size - 1) / page_size;
            if (avail_pages < need_pages) {
                LOG(DEBUG) << "need [" << need_pages << "] pages, but only [" << avail_pages << "] pages available";
                std::vector<int64_t> evicted_page_list;
                prefix_cache_mgr->Evict(need_pages - avail_pages, &evicted_page_list);
                page_mgr->Free(evicted_page_list.data(), evicted_page_list.size());
                if (int64_t(evicted_page_list.size()) < need_pages - avail_pages) {
                    prefix_cache_mgr->DecRefCount(check_res->hash_list.data(), check_res->hash_list.size());
                    return false;
                }
            }
            check_res->cache_hit_count = check_res->hash_list.size() * page_size;
            worker_profiler->step_counter.global.cache_hit_count += check_res->cache_hit_count;
            if (check_res->cache_hit_count != 0) {
                *is_prefix_cache_hit = true;
                LOG(INFO) << "Cache Hit [" << check_res->cache_hit_count << "]/[" << token_ids->size() << "] input tokens";
            }
            auto ret = page_mgr->Alloc(need_pages, &check_res->page_list);
            if (ret != RC_SUCCESS) {
                LOG(WARNING) << "[Nearly Impossible code] Alloc failed";
                prefix_cache_mgr->DecRefCount(check_res->hash_list.data(), check_res->hash_list.size());
                return false;
            }
            uint64_t token_pos = start;
            for (; token_pos < token_ids->size(); token_pos += page_size) {
                if (uint64_t(token_pos + page_size) > token_ids->size()) {
                    break;
                }
                uint64_t hash_val = utils::HashCombine(prev_hash, token_ids->data() + token_pos, page_size);
                int64_t page_id = check_res->page_list[token_pos / page_size];
                prefix_cache_mgr->Insert(hash_val, page_id);
                prev_hash = hash_val;
                check_res->hash_list.push_back(hash_val);
            }
            LOG(DEBUG) << "Do hash pages: " << (token_pos - start) / page_size << ", from [" << start << "] to [" << token_pos << "], total: " << token_ids->size();
        } else {
            int64_t page_num = (total_len + model_config.page_size - 1) / model_config.page_size;
            auto ret = page_mgr->Alloc(page_num, &check_res->page_list);
            if (ret != RC_SUCCESS) {
                return false;
            }

            LOG(INFO) << "task[" << req.orig->id << "] page_num: " << page_num;
        }
    }
    if (generator_config.enable_penalty) {
        check_res->slot_index = batch_slots_mgr->Alloc(1);
        if (check_res->slot_index == INT64_MAX) {
            LOG(ERROR) << "alloc batch slot error, alloc [" << 1 << "] but availale ["
                       << batch_slots_mgr->GetAvailableBlockNum() << "]";
            return false;
        }
    }

    return true;
}

void LLMGenerator::Generate() {
    RetCode rc;
    ModelInput model_input;
    ModelOutput model_output;
    int running_batch = 0;
    int prefill_batch = 0;
    long long loop_step = 0;
    int cache_cool_down_count = 0;
    std::string error_msg;
    RequestCheckResult check_res;
    std::vector<TidGenToken> tid_gen_token_list;
    map<int,int> decode_stat;
    map<int,vector<int>> decode_buffer;
    controller_.Reset();
    bool is_prefix_cache_hit = false;
    
    auto check_func = [this, &check_res, &cache_cool_down_count, &is_prefix_cache_hit](const LlmRequest& req) -> bool {
        check_res.cache_index = INT64_MAX;
        check_res.page_list.clear();
        check_res.slot_index = INT64_MAX;
        check_res.rest_iters = -1;
        check_res.first_fill_len = req.orig->token_ids->size();
        check_res.total_tokens_per_step += check_res.first_fill_len;
        check_res.hash_list.clear();
        check_res.cache_hit_count = 0;
        check_res.errmsg.clear();
        if (check_res.total_tokens_per_step > generator_config_.max_tokens_per_step) {
            return false;
        }

        if (!CheckTotalLen(generator_config_, req, &check_res)) {
            LOG(ERROR) << check_res.errmsg;
            return true;
        }

        if (!CheckAndAllocGPUMemory(model_config_, generator_config_, req, controller_.tid_list.size(), &idx_mgr_, &page_mgr_, &prefix_cache_mgr_,
                            &batch_slots_mgr_, &cache_cool_down_count, &check_res, worker_profiler_.get(), &is_prefix_cache_hit)) {
            return false;
        }

        check_res.running_batch++;
        check_res.prefill_batch++;
        return true;
    };

    std::map<uint64_t, TidData> tid_data_map;

#ifndef PPL_LLM_SERVING_SYNC_DECODE
    decoder_thread_pool_.RunAsync([](uint32_t, uint32_t) {});
#endif

    while (true) {
        is_prefix_cache_hit = false;
        auto global_start = std::chrono::high_resolution_clock::now();
        check_res.total_tokens_per_step = running_batch;
        check_res.running_batch = running_batch;
        check_res.prefill_batch = 0;
        {
            utils::TimingGuard __timing__(&worker_profiler_->step_counter.current.prepare_cost);

            while (true) {
                if (check_res.running_batch >= generator_config_.max_running_batch ||
                    check_res.prefill_batch >= generator_config_.max_prefill_batch ||
                    cache_cool_down_count > 0) {
                    break;
                }
                shared_ptr<LlmRequest> req(sched_.TryPopRequest(check_func));
                if (!req) {
                    break;
                }
                // queue waiting duration
                auto now = std::chrono::high_resolution_clock::now();

                ++worker_profiler_->req_counter.waiting_cnt;
                worker_profiler_->req_counter.waiting_cost +=
                    uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(now - req->enqueue_ts).count());

                if (!ParseRequest(*req, check_res, model_config_.cache_mode, &model_input, &controller_,
                                    &tid_data_map, conn_)) {
                    break;
                }
                controller_.req_list_changed = true;
            }
            running_batch = controller_.tid_list.size();
            if (running_batch == 0) {
                break;
            }

            UpdateInput(controller_, model_config_.cache_mode, controller_.req_list_changed, &model_input);
            worker_profiler_->max_running_task = std::max<uint64_t>(running_batch, worker_profiler_->max_running_task);
            prefill_batch = check_res.prefill_batch;
        }
        worker_profiler_->step_counter.global.prepare_cost += worker_profiler_->step_counter.current.prepare_cost;
        LOG(DEBUG) << "Step: " << loop_step << " ---------------------------------";

#ifdef PPL_LLM_ENABLE_DEBUG
        stringstream ss;
        for (int i = 0; i < running_batch; ++i) {
            ss << controller_.tid_list[i]->tid << ", ";
        }
        LOG(DEBUG) << "running tid: [" << ss.str() << "]";
        PrintVector(model_input.token_inputs, "token_inputs");
#endif
        model_output.Clear();
        model_output.Resize(running_batch);
        error_msg.clear();
        rc = llm_engine_.Execute(model_input, controller_.req_list_changed, is_prefix_cache_hit, &model_output, &error_msg);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "llm engine excute failed";
            for (auto* tid_ctrl : controller_.tid_list) {
                conn_->NotifyFailure(tid_ctrl->tid, rc, error_msg);
            }
            ReleaseResource();
            break;
        }
        controller_.req_list_changed = false; // reset

        // send stream chat rsp
        {
            utils::TimingGuard __timing__(&worker_profiler_->step_counter.current.post_process_cost);

#ifndef PPL_LLM_SERVING_SYNC_DECODE
            decoder_thread_pool_.Wait();
#endif
            tid_gen_token_list.clear();
            for (int task_iter = 0; task_iter < running_batch; ++task_iter) {
                auto* tid_ctrl = controller_.tid_list[task_iter];
                ++tid_ctrl->gen_tokens_cnt;
                int gen_token = model_output.output_token[task_iter];
                float logprob = model_output.logprobs[task_iter];
                LOG(DEBUG) << "task[" << tid_ctrl->tid << "] gen token: " << gen_token;

                // update params
                int32_t prev_seqlen = tid_ctrl->next_tokens->size();
                tid_ctrl->next_tokens.reset(new vector<int>({gen_token}));
                if (tid_ctrl->steps == 0) {
                    model_input.start_pos[task_iter] += prev_seqlen;
                    ++model_input.decoding_batches;
                } else {
                    model_input.start_pos[task_iter]++;
                }
                tid_ctrl->start_pos += prev_seqlen;
                tid_ctrl->steps++;
                tid_ctrl->rest_iters--;

                // finished task
                FinishFlag finish_flag = FinishFlag::NOT_FINISHED;
                if (tid_ctrl->rest_iters <= 0 ||
                    (tid_ctrl->early_stopping &&
                     ((generator_config_.stop_tokens.find(gen_token) != generator_config_.stop_tokens.end()) ||
                      (tid_ctrl->stop_tokens && tid_ctrl->stop_tokens->find(gen_token) != tid_ctrl->stop_tokens->end())))) {
                    finish_flag = tid_ctrl->rest_iters <= 0 ? FinishFlag::LENGTH : FinishFlag::EOS_TOKEN;

                    if (cache_cool_down_count > 0)
                        cache_cool_down_count--;
                    controller_.finished_tasks.Push(FinishedTaskInfo(tid_ctrl->tid, FinishedTaskInfo::FROM_WORKER));
                    controller_.req_list_changed = true;
                }
                bool is_special = generator_config_.special_tokens.find(gen_token) != generator_config_.special_tokens.end();
                tid_gen_token_list.emplace_back(tid_ctrl->tid, gen_token, logprob, finish_flag, tid_ctrl->steps,
                                                tid_ctrl->is_token_in_out, is_special);
            }

#ifndef PPL_LLM_SERVING_SYNC_DECODE
            decoder_thread_pool_.RunAsync([this, &tid_gen_token_list, &decode_stat, &decode_buffer](uint32_t nthr, uint32_t ithr) {
                size_t task_size = tid_gen_token_list.size();
                uint32_t n_block = task_size % nthr == 0 ? task_size / nthr : task_size / nthr + 1;
                uint32_t start_id = ithr * n_block;
                uint32_t end_id = (ithr + 1) * n_block > task_size ? task_size : (ithr + 1) * n_block;

                DecodeAndSendTask(start_id, end_id, tokenizer_, tid_gen_token_list, conn_, &decode_stat, &decode_buffer);
            });
#else
            DecodeAndSendTask(0, tid_gen_token_list.size(), tokenizer_, tid_gen_token_list, conn_, &decode_stat, &decode_buffer);
#endif
            // early finish
            if (controller_.finished_tasks.Size() > 0) {
                LOG(DEBUG) << "Do early finish";
                DeleteTasks(&model_input, &controller_.finished_tasks, &tid_data_map,
                            &worker_profiler_->finished_task_cnt, &worker_profiler_->req_counter.output_tokens_per_req);
            }
        }
        worker_profiler_->step_counter.global.post_process_cost +=
            worker_profiler_->step_counter.current.post_process_cost;

        auto global_end = std::chrono::high_resolution_clock::now();
        worker_profiler_->step_counter.current.total_cost =
            double(std::chrono::duration_cast<std::chrono::microseconds>(global_end - global_start).count());
        worker_profiler_->step_counter.global.total_cost += worker_profiler_->step_counter.current.total_cost;
        worker_profiler_->pending_task_size = sched_.GetPendingSize();
        worker_profiler_->step_counter.global.step_cnt++;
        ++loop_step;

        LOG(DEBUG) << "step: " << loop_step;

#ifdef PPL_LLM_ENABLE_PROFILING
        if (loop_step == 1 || loop_step % 100 == 0 || controller_.tid_list.size() == 0) {
            worker_profiler_->running_task = running_batch;
            worker_profiler_->prefill_batch = prefill_batch;
            worker_profiler_->prefill_tokens = model_input.token_inputs.size() - (running_batch - prefill_batch);
            worker_profiler_->kv_max_blk = kv_cache_max_tokens_;
            worker_profiler_->kv_rest_blk = model_config_.cache_mode == 0
                ? idx_mgr_.GetAvailableBlockNum()
                : page_mgr_.GetAvail() * model_config_.page_size;
#ifdef PPLNN_USE_LLM_CUDA
            cudaMemGetInfo(&worker_profiler_->dev_mem_free, &worker_profiler_->dev_mem_total);
#endif
            conn_->OnProfiling(worker_profiler_);
        }
#endif
    }

#ifndef PPL_LLM_SERVING_SYNC_DECODE
    decoder_thread_pool_.Wait();
#endif
}

void LLMGenerator::Process(const std::shared_ptr<Request>& req) {
    uint64_t encode_cost = 0;
    if (req->token_ids) {
        req->is_token_in_out = true;
    }
    {
        utils::TimingGuard __timing__(&encode_cost);
        if (!req->is_token_in_out) {
            req->token_ids = std::make_shared<std::vector<int>>();
            tokenizer_->Encode(req->prompt.data(), req->prompt.size(), req->token_ids.get());
            req->stop_tokens = std::make_shared<std::unordered_set<int>>();
            req->stop_tokens->insert(tokenizer_->GetEosId());
            conn_->OnTokenize(req->id, *req->token_ids);
        }
    }
    worker_profiler_->step_counter.global.input_token_cnt += req->token_ids->size();
    ++worker_profiler_->req_counter.encode_cnt;
    worker_profiler_->req_counter.encode_cost += encode_cost;

    auto* lreq = new LlmRequest();
    lreq->orig = req;
    lreq->enqueue_ts = std::chrono::high_resolution_clock::now();
    bool maybe_empty = sched_.PushRequest(lreq);
    if (maybe_empty) {
        req_signal_.NotifyOne();
    }
}

}} // namespace ppl::llm