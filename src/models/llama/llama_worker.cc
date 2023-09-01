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

#include "llama_worker.h"

#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#include "ppl/nn/common/logger.h"

#include <iostream>
#include <string>
#include <memory>
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <assert.h>
#include <omp.h>

using namespace std;

using namespace ppl::common;
using namespace ppl::nn;

#ifdef PPL_LLM_ENABLE_PROFILING
#include <cuda_runtime.h>
static void PrintMemUsage() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    float free = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    float used = total - free;
    fprintf(stderr, "memory usage: (%.2f - %.2f) -> %.2f GiB\n", total, free, used);
}
#endif

namespace ppl { namespace llm { namespace llama {

struct Profiler final {
    int prompt_cnt = 0;
    int gen_token_cnt = 0;

    int max_running_batch = 0;

    double prepare_duration = 0;
    double post_process_duration = 0;
    double model_duration = 0;
    double sampling_duration = 0;
    double total_duration = 0;
    double set_input_duration = 0;
    double send_duration = 0;
    double early_finish_duration = 0;

    double step_prepare_duration = 0;
    double step_post_process_duration = 0;
    double step_set_input_duration = 0;
    double step_model_duration = 0;
    double step_sampling_duration = 0;
    double step_send_duration = 0;
    double step_early_finish_duration = 0;
    double step_total_duration = 0;
};

struct TidGenTokens final {
    TidGenTokens(uint64_t tid, const std::vector<int>& gen_tokens) : tid(tid), gen_tokens(gen_tokens) {}

    uint64_t tid;
    std::vector<int> gen_tokens;
};

class DecoderThreadTask final : public ppl::common::ThreadTask {
public:
    DecoderThreadTask(const sentencepiece::SentencePieceProcessor* tokenizer,
                      std::unordered_map<uint64_t, LLaMAWorker::UuidData>* uuid_data, pthread_mutex_t* uuid_data_lock,
                      const std::shared_ptr<std::unordered_map<uint64_t, std::vector<int>>>& tid_gen_tokens,
                      const std::shared_ptr<std::vector<TidGenTokens>>& last_tid_gen_tokens,
                      pthread_mutex_t* decoder_lock)
        : uuid_data_(uuid_data)
        , uuid_data_lock_(uuid_data_lock)
        , tokenizer_(tokenizer)
        , tid_gen_tokens_(tid_gen_tokens)
        , last_tid_gen_tokens_(last_tid_gen_tokens)
        , decoder_lock_(decoder_lock) {}

    std::shared_ptr<ppl::common::ThreadTask> Run() override;

private:
    void FindUData(uint64_t uuid, LLaMAWorker::UuidData* udata, bool should_remove) {
        pthread_mutex_lock(uuid_data_lock_);
        auto uuid_data_ref = uuid_data_->find(uuid);
        if (uuid_data_ref != uuid_data_->end()) {
            *udata = uuid_data_ref->second;
            if (should_remove) {
                uuid_data_->erase(uuid_data_ref);
            }
        }
        pthread_mutex_unlock(uuid_data_lock_);
    }

private:
    std::unordered_map<uint64_t, LLaMAWorker::UuidData>* uuid_data_;
    pthread_mutex_t* uuid_data_lock_;
    const sentencepiece::SentencePieceProcessor* tokenizer_;
    std::shared_ptr<std::unordered_map<uint64_t, std::vector<int>>> tid_gen_tokens_;
    std::shared_ptr<std::vector<TidGenTokens>> last_tid_gen_tokens_;
    pthread_mutex_t* decoder_lock_;
};

std::shared_ptr<ppl::common::ThreadTask> DecoderThreadTask::Run() {
    size_t running_batch = tid_gen_tokens_->size() + last_tid_gen_tokens_->size();

    #pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < running_batch; ++i) {
        Response rsp;
        if (i < tid_gen_tokens_->size()) {
            auto it = tid_gen_tokens_->begin();
            std::advance(it, i);
            auto tid = it->first;
            auto& gen_tokens = it->second;
            int last_gen_token = gen_tokens.back();
            const std::string& cur_piece = tokenizer_->IdToPiece(last_gen_token);

            if (gen_tokens.size() != 1 && cur_piece.substr(0, 3) == "▁") { // normal case
                std::vector<int> new_tokens(gen_tokens.begin(), gen_tokens.end() - 1);
                tokenizer_->Decode(new_tokens, &rsp.generated);
                LOG(DEBUG) << "task[" << tid << "] new word: " << rsp.generated;
                gen_tokens[0] = gen_tokens.back();
                gen_tokens.resize(1);
                // send response
                LLaMAWorker::UuidData udata;
                FindUData(tid, &udata, false);
                if (udata.conn) {
                    rsp.id = udata.req_id;
                    rsp.flag = Response::NORMAL;
                    udata.conn->Send(rsp);
                }
            }
        } else { // send last word
            auto& tid_info = last_tid_gen_tokens_->at(i - tid_gen_tokens_->size());
            auto tid = tid_info.tid;
            auto& gen_tokens = tid_info.gen_tokens;

            tokenizer_->Decode(gen_tokens, &rsp.generated);
            LOG(DEBUG) << "task[" << tid << "] last word: " << rsp.generated;

            LLaMAWorker::UuidData udata;
            FindUData(tid, &udata, true);
            if (udata.conn) {
                rsp.id = udata.req_id;
                rsp.flag = Response::IS_LAST;
                udata.conn->Send(rsp);
            }
        }
    }

    pthread_mutex_unlock(decoder_lock_);
    return shared_ptr<ThreadTask>();
}

#ifdef PPL_LLM_ENABLE_PROFILING
static void PrintProfilingMsg(const Profiler& profiler, int step, int running_batch, uint64_t kv_max_blk,
                              uint64_t kv_rest_blk) {
    fprintf(stderr, "[PERF] --- step %d -------------------------------------------------\n", step);
    fprintf(stderr, "[PERF]  |- ");
    ::PrintMemUsage();
    fprintf(stderr, "[PERF]  |- kv cache usage: %.2f %%\n", (1.0f - (double)kv_rest_blk / kv_max_blk) * 100.0);
    fprintf(stderr, "[PERF]  |- running batch: %d, max running batch: %d\n", running_batch, profiler.max_running_batch);
    fprintf(stderr, "[PERF]  |- finished query count: %d, QPS: %.2f\n", profiler.prompt_cnt,
            float(profiler.prompt_cnt) / profiler.total_duration * 1000);
    fprintf(stderr, "[PERF]  |- gen token count: %d, avg gen len: %.2f, TPS: %.2f\n", profiler.gen_token_cnt,
            profiler.prompt_cnt ? profiler.gen_token_cnt / float(profiler.prompt_cnt) : 0.0f,
            float(profiler.gen_token_cnt) / profiler.total_duration * 1000);

    fprintf(stderr, "[PERF]  |- pipeline          | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_total_duration, profiler.total_duration / step, profiler.total_duration);
    fprintf(stderr, "[PERF]  |-- batching         | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_prepare_duration, profiler.prepare_duration / step, profiler.prepare_duration);
    fprintf(stderr, "[PERF]  |-- copy inputs      | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_set_input_duration, profiler.set_input_duration / step, profiler.set_input_duration);
    fprintf(stderr, "[PERF]  |-- model inference  | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_model_duration, profiler.model_duration / step, profiler.model_duration);
    fprintf(stderr, "[PERF]  |-- sampling         | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_sampling_duration, profiler.sampling_duration / step, profiler.sampling_duration);
    fprintf(stderr, "[PERF]  |-- increase next    | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_post_process_duration, profiler.post_process_duration / step, profiler.post_process_duration);
    fprintf(stderr, "[PERF]  |-- send response    | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_send_duration, profiler.send_duration / step, profiler.send_duration);
    fprintf(stderr, "[PERF]  |-- early finish     | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_early_finish_duration, profiler.early_finish_duration / step, profiler.early_finish_duration);

    fprintf(stderr, "[PERF]  |- schedule cost: %.2f %%\n",
            (profiler.total_duration - profiler.model_duration) / profiler.total_duration * 100);
}
#endif

#ifdef PPL_LLM_ENABLE_DEBUG
template <class T>
static void PrintVector(std::vector<T> vec) {
    for (auto& ele : vec) {
        std::cout << ele << ", ";
    }
    std::cout << std::endl;
}

static string GetDimsStr(const Tensor* tensor) {
    auto shape = tensor->GetShape();
    if (shape->GetRealDimCount() == 0) {
        return string();
    }

    string res = ToString(shape->GetDim(0));
    for (uint32_t i = 1; i < shape->GetDimCount(); ++i) {
        res += "_" + ToString(shape->GetDim(i));
    }

    return res;
}

static const pair<string, datatype_t> g_str2datatype[] = {
    {"fp64", DATATYPE_FLOAT64}, {"fp32", DATATYPE_FLOAT32}, {"fp16", DATATYPE_FLOAT16}, {"int32", DATATYPE_INT32},
    {"int64", DATATYPE_INT64},  {"int8", DATATYPE_INT8},    {"bool", DATATYPE_BOOL},    {"", DATATYPE_UNKNOWN},
};

static const char* FindDataTypeStr(datatype_t dt) {
    for (int i = 0; !g_str2datatype[i].first.empty(); ++i) {
        if (g_str2datatype[i].second == dt) {
            return g_str2datatype[i].first.c_str();
        }
    }
    return nullptr;
}

static bool SaveOutputsOneByOne(const Runtime* runtime, const std::string& tag = "") {
    std::string g_flag_save_data_dir = ".";
    for (uint32_t c = 0; c < runtime->GetOutputCount(); ++c) {
        auto t = runtime->GetOutputTensor(c);

        ppl::nn::TensorShape dst_desc = *t->GetShape();
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        // convert fp16 to fp32
        if (dst_desc.GetDataType() == DATATYPE_FLOAT16) {
            dst_desc.SetDataType(DATATYPE_FLOAT32);
        }

        auto bytes = dst_desc.CalcBytesIncludingPadding();
        vector<char> buffer(bytes);
        auto status = t->ConvertToHost(buffer.data(), dst_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data of tensor[" << t->GetName() << "] failed: " << GetRetCodeStr(status);
            return false;
        }

        const string out_file_name =
            g_flag_save_data_dir + "/" + "step_" + tag + "pplnn_output-" + t->GetName() + ".dat";
        ofstream ofs(out_file_name, ios_base::out | ios_base::binary | ios_base::trunc);
        if (!ofs.is_open()) {
            LOG(ERROR) << "open output file[" << out_file_name << "]";
            return false;
        }

        ofs.write(buffer.data(), bytes);
    }
    return true;
}
static bool SaveInputsOneByOne(const Runtime* runtime, const std::string& tag = "") {
    std::string g_flag_save_data_dir = ".";

    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto shape = t->GetShape();

        auto bytes = shape->CalcBytesIncludingPadding();
        vector<char> buffer(bytes);

        ppl::nn::TensorShape src_desc = *t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        if (std::string(t->GetName()) == "kv_cache" || std::string(t->GetName()) == "kv_scale") {
            continue;
        }

        LOG(INFO) << t->GetName();
        auto status = t->ConvertToHost(buffer.data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert tensor[" << t->GetName() << "] content failed: " << GetRetCodeStr(status);
            continue;
        }

        const char* data_type_str = FindDataTypeStr(shape->GetDataType());
        if (!data_type_str) {
            LOG(ERROR) << "unsupported data type[" << GetDataTypeStr(shape->GetDataType()) << "]";
            return false;
        }

        char name_prefix[32];
        sprintf(name_prefix, "pplnn_input%s_%05u_", tag.c_str(), c);
        const string in_file_name = g_flag_save_data_dir + "/" + "step_" + tag + string(name_prefix) + t->GetName() +
            "-" + GetDimsStr(t) + "-" + string(data_type_str) + ".dat";
        ofstream ofs(in_file_name, ios_base::out | ios_base::binary | ios_base::trunc);
        if (!ofs.is_open()) {
            LOG(ERROR) << "save input file[" << in_file_name << "] failed.";
            return false;
        }

        ofs.write(buffer.data(), bytes);
    }

    return true;
}
#endif

LLaMAWorker::LLaMAWorker(const sentencepiece::SentencePieceProcessor* tokenizer, const Resource& resource,
                         const ModelConfig& mconfig, const WorkerConfig& wconfig)
    : tokenizer_(tokenizer)
    , model_config_(mconfig)
    , worker_config_(wconfig)
    , tensor_parallel_size_(resource.tensor_parallel_size)
    , worker_thread_args_(resource.tensor_parallel_size) {
    pthread_mutex_init(&uuid_data_lock_, nullptr);

    if (model_config_.auto_causal != true) {
        LOG(ERROR) << "only support auto_causal == true";
        exit(-1);
    }

    if (model_config_.cache_layout != 0 && model_config_.cache_mode != 0) {
        LOG(ERROR) << "only support cache_layout == 0 and cache_mode == 0";
        exit(-1);
    }

    if (model_config_.cache_quant_bit != 8 && model_config_.cache_quant_group != 8) {
        LOG(ERROR) << "only support cache_quant_bit == 8 and cache_quant_group == 8";
        exit(-1);
    }

    if (model_config_.dynamic_batching != true) {
        LOG(ERROR) << "only support dynamic_batching == true";
        exit(-1);
    }

    kv_cache_max_tokens_ = resource.kv_cache_max_tokens;

    idx_mgr_.Init(kv_cache_max_tokens_);

    for (int i = 0; i < tensor_parallel_size_; i++) {
        auto arg = &worker_thread_args_[i];
        arg->resource = &resource.items[i];

        arg->token_ids = arg->resource->runtime->GetInputTensor(0);
        arg->attn_mask = arg->resource->runtime->GetInputTensor(1);
        arg->seq_starts = arg->resource->runtime->GetInputTensor(2);
        arg->kv_starts = arg->resource->runtime->GetInputTensor(3);
        arg->cache_indices = arg->resource->runtime->GetInputTensor(4);
        arg->decoding_batches = arg->resource->runtime->GetInputTensor(5);
        arg->start_pos = arg->resource->runtime->GetInputTensor(6);
        arg->max_seq_len = arg->resource->runtime->GetInputTensor(7);
        arg->max_kv_len = arg->resource->runtime->GetInputTensor(8);
        arg->kv_cache = arg->resource->runtime->GetInputTensor(9);
        arg->kv_scale = arg->resource->runtime->GetInputTensor(10);

        arg->logits = arg->resource->runtime->GetOutputTensor(0);

        arg->decoding_batches->SetDeviceContext(arg->resource->runtime->GetHostDeviceContext());
        arg->max_seq_len->SetDeviceContext(arg->resource->runtime->GetHostDeviceContext());
        arg->max_kv_len->SetDeviceContext(arg->resource->runtime->GetHostDeviceContext());

        arg->kv_cache->GetShape()->Reshape({(int64_t)kv_cache_max_tokens_, model_config_.num_layers, 2,
                                            model_config_.num_kv_heads / tensor_parallel_size_,
                                            model_config_.hidden_dim / model_config_.num_heads});
        arg->kv_scale->GetShape()->Reshape(
            {(int64_t)kv_cache_max_tokens_, model_config_.num_layers, 2,
             model_config_.num_kv_heads / tensor_parallel_size_,
             model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});

        arg->kv_cache->SetBufferPtr(arg->resource->kv_cache_mem);
        arg->kv_scale->SetBufferPtr(arg->resource->kv_scale_mem);
    }

    pthread_mutex_init(&decoder_lock_, nullptr);
}

LLaMAWorker::~LLaMAWorker() {
    pthread_mutex_destroy(&uuid_data_lock_);
    pthread_mutex_destroy(&decoder_lock_);
    pthread_cond_destroy(&req_signal_);
    pthread_mutex_destroy(&lock_for_req_signal_);
}

RetCode LLaMAWorker::Init() {
    auto ret = thread_pool_.Init(1);
    if (ret != RC_SUCCESS) {
        LOG(ERROR) << "Init Thread Pool error";
        return RC_OTHER_ERROR;
    }

    pthread_mutex_init(&lock_for_req_signal_, nullptr);
    pthread_cond_init(&req_signal_, nullptr);
    return RC_SUCCESS;
}

bool LLaMAWorker::ParseRequest(const LlamaRequest& req, std::unordered_map<uint64_t, TidController>* tid_controllers) {
    if (check_result_.rest_iters < 0) {
        req.conn->NotifyFailure(req.orig->id);
        return true;
    }

    if (check_result_.cache_index == INT64_MAX) {
        LOG(ERROR) << "catch invalid cache_index.";
        return false;
    }

    auto tid = req.uuid;
    auto& tid_ctrl = tid_controllers->emplace(tid, TidController()).first->second;
    tid_ctrl.tid = tid;
    tid_ctrl.prompt = req.orig->prompt;
    tid_ctrl.temperature = req.orig->temperature;
    tid_ctrl.rest_iters = check_result_.rest_iters;
    tid_ctrl.first_fill_len = check_result_.first_fill_len;
    tid_ctrl.total_len = check_result_.first_fill_len + check_result_.rest_iters;
    tid_ctrl.cache_index = check_result_.cache_index;
    tid_ctrl.next_tokens = std::move(req.token_id_list);

    worker_controller_.tid_list.push_back(&tid_ctrl);
    worker_controller_.start_pos.push_back(0);
    worker_controller_.temperatures.push_back(tid_ctrl.temperature);
    worker_controller_.cache_indices.push_back(check_result_.cache_index);
    worker_controller_.total_rest_iters = std::max<int64_t>(worker_controller_.total_rest_iters, tid_ctrl.rest_iters);

    return true;
}

void LLaMAWorker::ClearTask(Connection* conn) {
    pthread_mutex_lock(&uuid_data_lock_);
    auto ref = conn2uuid_.find(conn);
    if (ref == conn2uuid_.end()) {
        pthread_mutex_unlock(&uuid_data_lock_);
        return;
    }
    auto conn_uuid_list = ref->second;
    for (auto x = ref->second.begin(); x != ref->second.end(); ++x) {
        uuid_data_.erase(*x);
    }
    conn2uuid_.erase(ref);
    pthread_mutex_unlock(&uuid_data_lock_);
}

RetCode LLaMAWorker::SetInputsTensor(WorkerThreadArg& arg) {
    RetCode rc;

    // 0. token ids
    // LOG(INFO) << "token inputs";
    // PrintVector(worker_controller_.token_inputs);
    int64_t* token_ids_data = worker_controller_.token_inputs.data();
    arg.token_ids->GetShape()->Reshape({int64_t(worker_controller_.token_inputs.size())});
    rc = arg.token_ids->CopyFromHostAsync(token_ids_data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set token_ids [" << arg.token_ids->GetName() << "] failed: " << GetRetCodeStr(rc);
        return RC_OTHER_ERROR;
    }

    // 2. seq_start
    // LOG(INFO) << "seq_starts";
    // PrintVector(worker_controller_.seq_starts);
    int64_t* seq_starts_data = worker_controller_.seq_starts.data();
    arg.seq_starts->GetShape()->Reshape({int64_t(worker_controller_.seq_starts.size())});
    rc = arg.seq_starts->CopyFromHostAsync(seq_starts_data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set seq_starts [" << arg.seq_starts->GetName() << "] failed: " << GetRetCodeStr(rc);
        return RC_OTHER_ERROR;
    }

    // 3. kv_starts
    // LOG(INFO) << "kv_starts";
    // PrintVector(worker_controller_.kv_starts);
    int64_t* kv_starts_data = worker_controller_.kv_starts.data();
    arg.kv_starts->GetShape()->Reshape({int64_t(worker_controller_.kv_starts.size())});
    rc = arg.kv_starts->CopyFromHostAsync(kv_starts_data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set kv_starts " << arg.kv_starts->GetName() << " failed: " << GetRetCodeStr(rc);
        return RC_OTHER_ERROR;
    }

    // 4. cache_indices
    // LOG(INFO) << "cache_indices";
    // PrintVector(worker_controller_.cache_indices);
    int64_t* cache_indices_data = worker_controller_.cache_indices.data();
    arg.cache_indices->GetShape()->Reshape({int64_t(worker_controller_.cache_indices.size())});
    rc = arg.cache_indices->CopyFromHostAsync(cache_indices_data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set cache_indices [" << arg.cache_indices->GetName() << "] failed: " << GetRetCodeStr(rc);
        return RC_OTHER_ERROR;
    }

    // 5. decoding batches
    // LOG(INFO) << "decoding_batches: " << worker_controller_.decoding_batches;
    int64_t* decoding_batches_data = &worker_controller_.decoding_batches;
    rc = arg.decoding_batches->CopyFromHostAsync(decoding_batches_data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set decoding_batches [" << arg.decoding_batches->GetName() << "] failed: " << GetRetCodeStr(rc);
        return RC_OTHER_ERROR;
    }

    // 6. start_pos
    // LOG(INFO) << "start_pos";
    // PrintVector(worker_controller_.start_pos);
    int64_t* start_pos_data = worker_controller_.start_pos.data();
    arg.start_pos->GetShape()->Reshape({int64_t(worker_controller_.start_pos.size())});
    rc = arg.start_pos->CopyFromHostAsync(start_pos_data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set start_pos [" << arg.start_pos->GetName() << "] failed: " << GetRetCodeStr(rc);
        return RC_OTHER_ERROR;
    }

    // 7. max_seq_len
    // LOG(INFO) << "max_seq_len: " << worker_controller_.max_seq_len;
    int64_t* max_seq_len_data = &worker_controller_.max_seq_len;
    rc = arg.max_seq_len->CopyFromHostAsync(max_seq_len_data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set max_seq_len [" << arg.max_seq_len->GetName() << "] failed: " << GetRetCodeStr(rc);
        return RC_OTHER_ERROR;
    }

    // 8. max_kv_len
    // LOG(INFO) << "max_kv_len: " << worker_controller_.max_kv_len;
    int64_t* max_kv_len_data = &worker_controller_.max_kv_len;
    rc = arg.max_kv_len->CopyFromHostAsync(max_kv_len_data);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set max_kv_len [" << arg.max_kv_len->GetName() << "] failed: " << GetRetCodeStr(rc);
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

static int RemoveFinishedTask(WorkerController* worker_controller, void* ptr) {
    int running_batch = worker_controller->tid_list.size();
    int left = 0;
    for (int right = 0; right < running_batch; right++) {
        if (worker_controller->tid_list[right] != nullptr) {
            worker_controller->tid_list[left] = worker_controller->tid_list[right];
            worker_controller->cache_indices[left] = worker_controller->cache_indices[right];
            worker_controller->start_pos[left] = worker_controller->start_pos[right];
            worker_controller->temperatures[left] = worker_controller->temperatures[right];
            ++left;
        }
    }
    worker_controller->tid_list.resize(left);
    worker_controller->cache_indices.resize(left);
    worker_controller->start_pos.resize(left);
    worker_controller->temperatures.resize(left);
    LOG(DEBUG) << "Rest tasks: " << worker_controller->tid_list.size();
    return left;
}

void LLaMAWorker::DeleteTask(const std::vector<uint64_t>& finished_list,
                             std::unordered_map<uint64_t, TidController>* tid_controllers) {
    // process finished task
    for (size_t i = 0; i < finished_list.size(); ++i) {
        auto tid = finished_list[i];
        auto tid_it = tid_controllers->find(tid);
        if (tid_it == tid_controllers->end()) {
            LOG(ERROR) << "find non exist tid: " << tid;
            continue;
        }
        auto& tid_ctrl = tid_it->second;

        if (tid_ctrl.is_first_fill == false) { // avoid corner case generation_len = 0
            --worker_controller_.decoding_batches;
        }
        // search deleted element
        size_t task_iter = 0;
        for (; task_iter < worker_controller_.tid_list.size(); ++task_iter) {
            if (worker_controller_.tid_list[task_iter] != nullptr &&
                worker_controller_.tid_list[task_iter]->tid == tid) {
                break;
            }
        }
        if (task_iter == worker_controller_.tid_list.size()) {
            LOG(ERROR) << "Delete not exist task";
            continue;
        }

        worker_controller_.tid_list[task_iter] = nullptr;
        worker_controller_.cache_indices[task_iter] = INT64_MAX;
        worker_controller_.start_pos[task_iter] = INT64_MAX;
        worker_controller_.temperatures[task_iter] = 0.0f;

        idx_mgr_.Free(tid_ctrl.cache_index, tid_ctrl.total_len);
        tid_controllers->erase(tid_it);
    }
    // update rest iter, for eos and shutdown finished
    worker_controller_.total_rest_iters = 0;
    for (size_t i = 0; i < worker_controller_.tid_list.size(); i++) {
        if (worker_controller_.tid_list[i] == nullptr) {
            continue;
        }
        worker_controller_.total_rest_iters =
            std::max<int>(worker_controller_.total_rest_iters, worker_controller_.tid_list[i]->rest_iters);
    }
    RemoveFinishedTask(&worker_controller_, nullptr);
}

static void UpdateInput(const std::unordered_map<uint64_t, TidController>& tid_controllers,
                        WorkerController* worker_controller) {
    // update input
    int running_batch = worker_controller->tid_list.size();

    worker_controller->max_seq_len = 0;
    worker_controller->max_kv_len = 0;
    worker_controller->token_inputs.clear();
    worker_controller->seq_starts.clear();
    worker_controller->seq_starts.reserve(running_batch + 1);
    worker_controller->seq_starts.push_back(0);

    worker_controller->kv_starts.clear();
    worker_controller->kv_starts.reserve(running_batch + 1);
    worker_controller->kv_starts.push_back(0);

    worker_controller->tid_finished.clear();

    for (int i = 0; i < running_batch; ++i) {
        auto* tid_ctrl = worker_controller->tid_list[i];

        worker_controller->token_inputs.insert(worker_controller->token_inputs.end(), tid_ctrl->next_tokens.begin(),
                                               tid_ctrl->next_tokens.end());
        worker_controller->seq_starts.push_back(worker_controller->seq_starts[i] + tid_ctrl->next_tokens.size());
        worker_controller->kv_starts.push_back(worker_controller->kv_starts[i] + tid_ctrl->first_fill_len +
                                               tid_ctrl->passed_iters);

        worker_controller->max_seq_len =
            std::max<int64_t>(tid_ctrl->next_tokens.size(), worker_controller->max_seq_len);
        worker_controller->max_kv_len =
            std::max<int64_t>(tid_ctrl->first_fill_len + tid_ctrl->passed_iters, worker_controller->max_kv_len);
    }
}

void LLaMAWorker::Work() {
    Profiler profiler;
    worker_controller_.Reset();

    std::unordered_map<uint64_t, TidController> tid_controllers;
    auto tid_gen_tokens = std::make_shared<std::unordered_map<uint64_t, std::vector<int>>>();

    long long step = 0;
    int cache_cool_down_count = 0;

    std::shared_ptr<LlamaRequest> req;

    auto check_func = [this, &cache_cool_down_count](const LlamaRequest& req) -> bool {
        RequestCheckResult& res = check_result_;

        res.cache_index = INT64_MAX;
        res.rest_iters = -1;
        res.first_fill_len = req.token_id_list.size();

        if (res.first_fill_len + req.orig->generation_length > (size_t)worker_config_.max_tokens_per_request) {
            res.rest_iters = worker_config_.max_tokens_per_request - res.first_fill_len;
        } else {
            res.rest_iters = req.orig->generation_length;
        }

        if (res.rest_iters < 0) {
            return true;
        }

        res.cache_index = idx_mgr_.Alloc(res.first_fill_len + res.rest_iters);

        if (res.cache_index == INT64_MAX) {
            cache_cool_down_count = std::min(std::max(1, (int)floorf(worker_controller_.tid_list.size() * 0.1f)), 8);
            return false;
        }

        return true;
    };

    while (true) {
        // 1. Recv and Parse Requset
        auto global_start = std::chrono::high_resolution_clock::now();

        auto local_start = std::chrono::high_resolution_clock::now();
        if (worker_controller_.tid_list.size() < (size_t)worker_config_.max_running_batch &&
            cache_cool_down_count <= 0) {
            while (true) {
                req = sched_.TryPopRequest(check_func);
                if (!req) {
                    break;
                }

                if (!ParseRequest(*req, &tid_controllers)) {
                    break;
                }

                if (worker_controller_.tid_list.size() >= (size_t)worker_config_.max_running_batch) {
                    break;
                }
            }

            if (worker_controller_.total_rest_iters == 0) {
                break;
            }
        }

        int running_batch = worker_controller_.tid_list.size();
        if (running_batch == 0) {
            break;
        }

        UpdateInput(tid_controllers, &worker_controller_);

        profiler.max_running_batch = std::max(running_batch, profiler.max_running_batch);
        auto local_end = std::chrono::high_resolution_clock::now();
        profiler.step_prepare_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(local_end - local_start).count()) / 1000.0;
        profiler.prepare_duration += profiler.step_prepare_duration;

        LOG(DEBUG) << "Step: " << step << " ---------------------------------";

        // 2. set inputs tensor
        local_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(tensor_parallel_size_)
        {
            int thread_id = omp_get_thread_num();
            auto rc = SetInputsTensor(worker_thread_args_[thread_id]);
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "thread[" << thread_id << "]"
                           << " SetInputsTensor failed: " << GetRetCodeStr(rc);
                exit(-1);
            }
            worker_thread_args_[thread_id].resource->runtime->Synchronize();
        }
        local_end = std::chrono::high_resolution_clock::now();
        profiler.step_set_input_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(local_end - local_start).count()) / 1000.0;
        profiler.set_input_duration += profiler.step_set_input_duration;

        // 3. model forward
        local_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel num_threads(tensor_parallel_size_)
        {
            int thread_id = omp_get_thread_num();
            RetCode rc;
            // rc = SetInputsTensor(worker_thread_args_[thread_id]);
            // if (rc != RC_SUCCESS) {
            //     LOG(ERROR) << "thread[" << thread_id << "]" << " SetInputsTensor failed: " << GetRetCodeStr(rc);
            //     exit(-1);
            // }
            rc = worker_thread_args_[thread_id].resource->runtime->Run(); // forward
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "thread[" << thread_id << "]"
                           << " runtime->Run() failed: " << GetRetCodeStr(rc);
                exit(-1);
            }
        }
        local_end = std::chrono::high_resolution_clock::now();
        profiler.step_model_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(local_end - local_start).count()) / 1000.0;
        profiler.model_duration += profiler.step_model_duration;

        // 4. sampling
        local_start = std::chrono::high_resolution_clock::now();
        auto logits = worker_thread_args_[0].logits;
        std::vector<int32_t> gen_tokens(running_batch);
        {
            RetCode rc = sampler_->SampleTopPTopK(
                (float*)logits->GetBufferPtr(), worker_controller_.temperatures.data(), running_batch,
                model_config_.vocab_size, worker_config_.top_p, worker_config_.top_k, gen_tokens.data());
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "SampleTopPTopK failed: " << GetRetCodeStr(rc);
                exit(-1);
            }
        }
        local_end = std::chrono::high_resolution_clock::now();
        profiler.sampling_duration +=
            double(std::chrono::duration_cast<std::chrono::microseconds>(local_end - local_start).count()) / 1000.0;
        profiler.step_sampling_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(local_end - local_start).count()) / 1000.0;

        // 5. post process
        local_start = std::chrono::high_resolution_clock::now();
        profiler.gen_token_cnt += running_batch;
        for (int task_iter = 0; task_iter < running_batch; ++task_iter) {
            auto* tid_ctrl = worker_controller_.tid_list[task_iter];

            int gen_token = gen_tokens[task_iter];
            LOG(DEBUG) << "task[" << tid_ctrl->tid << "] gen token: " << gen_token;

            // update tid controller state:
            tid_ctrl->next_tokens = {gen_token};
            // update input, worker controller state: start pos, decoding batches
            if (tid_ctrl->is_first_fill == true) {
                worker_controller_.start_pos[task_iter] += tid_ctrl->first_fill_len;
                tid_ctrl->is_first_fill = false;
                worker_controller_.decoding_batches += 1;
            } else {
                worker_controller_.start_pos[task_iter]++;
            }

            // update params
            tid_ctrl->passed_iters++;
            tid_ctrl->rest_iters--;

            // 检测finish: if rest_iters为0或者收到end token
            if (tid_ctrl->rest_iters <= 0 || gen_token == tokenizer_->eos_id()) {
                if (cache_cool_down_count > 0)
                    cache_cool_down_count--;
                worker_controller_.tid_finished.push_back(tid_ctrl->tid);
            }
        }
        local_end = std::chrono::high_resolution_clock::now();
        profiler.step_post_process_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(local_end - local_start).count()) / 1000.0;
        profiler.post_process_duration += profiler.step_post_process_duration;

        // 6. send stream chat rsp
        local_start = std::chrono::high_resolution_clock::now();
        pthread_mutex_lock(&decoder_lock_);
        auto last_tid_gen_tokens = std::make_shared<std::vector<TidGenTokens>>();
        for (int task_iter = 0; task_iter < running_batch; ++task_iter) {
            auto* tid_ctrl = worker_controller_.tid_list[task_iter];
            int gen_token = gen_tokens[task_iter];
            auto iter = tid_gen_tokens->emplace(tid_ctrl->tid, std::vector<int>()).first;
            iter->second.push_back(gen_token);

            // finished task
            if (tid_ctrl->rest_iters <= 0 || gen_token == tokenizer_->eos_id()) {
                last_tid_gen_tokens->emplace_back(TidGenTokens(tid_ctrl->tid, iter->second));
                tid_gen_tokens->erase(iter);
            }
        }

        auto rc = thread_pool_.AddTask(std::make_shared<DecoderThreadTask>(
            tokenizer_, &uuid_data_, &uuid_data_lock_, tid_gen_tokens, last_tid_gen_tokens, &decoder_lock_));
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "thread_pool_.AddTask() failed: " << GetRetCodeStr(rc);
            exit(-1);
        }
        local_end = std::chrono::high_resolution_clock::now();
        profiler.step_send_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(local_end - local_start).count()) / 1000.0;
        profiler.send_duration += profiler.step_send_duration;

        local_start = std::chrono::high_resolution_clock::now();
        // update iteration
        step++;
        worker_controller_.total_rest_iters--;
        LOG(DEBUG) << "Rest iters: " << worker_controller_.total_rest_iters;

        // 6. early finish
        if (worker_controller_.tid_finished.size() > 0) {
            LOG(DEBUG) << "Do early finish";
            profiler.prompt_cnt += worker_controller_.tid_finished.size();
            DeleteTask(worker_controller_.tid_finished, &tid_controllers);
        }
        local_end = std::chrono::high_resolution_clock::now();
        profiler.step_early_finish_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(local_end - local_start).count()) / 1000.0;
        profiler.early_finish_duration += profiler.step_early_finish_duration;

        auto global_end = std::chrono::high_resolution_clock::now();
        profiler.step_total_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(global_end - global_start).count()) / 1000.0;
        profiler.total_duration += profiler.step_total_duration;

#ifdef PPL_LLM_ENABLE_PROFILING
        if (step % 100 == 0 || worker_controller_.total_rest_iters == 0) {
            PrintProfilingMsg(profiler, step, running_batch, kv_cache_max_tokens_, idx_mgr_.GetAvailableBlockNum());
        }
#endif
    }
}

void LLaMAWorker::Process(const shared_ptr<Request>& req, Connection* conn) {
    auto lreq = make_shared<LlamaRequest>();
    lreq->uuid = uuid_seq_++;
    lreq->conn = conn;
    lreq->orig = req;
    lreq->token_id_list.push_back(tokenizer_->bos_id());
    tokenizer_->Encode(req->prompt, &lreq->token_id_list);

    pthread_mutex_lock(&uuid_data_lock_);
    uuid_data_.insert(make_pair(lreq->uuid, UuidData(req->id, conn)));
    auto ret_pair = conn2uuid_.insert(make_pair(conn, vector<uint64_t>()));
    ret_pair.first->second.push_back(lreq->uuid);
    pthread_mutex_unlock(&uuid_data_lock_);

    sched_.PushRequest(lreq);
    pthread_cond_signal(&req_signal_);
}

void LLaMAWorker::Wait() {
    pthread_mutex_lock(&lock_for_req_signal_);
    LOG(INFO) << "waiting for request ...";
    pthread_cond_wait(&req_signal_, &lock_for_req_signal_);
    pthread_mutex_unlock(&lock_for_req_signal_);
}

}}} // namespace ppl::llm::llama
