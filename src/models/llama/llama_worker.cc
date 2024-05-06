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
#include "../../utils/utils.h"

#include "ppl/nn/common/logger.h"

#include <string>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>

#ifdef PPL_LLM_ENABLE_DEBUG
#include <iostream>
#include <fstream>
#endif

using namespace std;

using namespace ppl::common;
using namespace ppl::nn;

#ifdef PPL_LLM_ENABLE_PROFILING
#ifdef PPLNN_USE_LLM_CUDA
#include <cuda_runtime.h>
static void PrintMemUsage() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    float free = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    float used = total - free;
    fprintf(stderr, "memory usage: (%.2f - %.2f) -> %.2f GiB\n", total, free, used);
}
#else
static void PrintMemUsage() {
    fprintf(stderr, "memory usage: unknown\n");
}
#endif
#endif

namespace ppl { namespace llm { namespace llama {

struct Profiler final {
    int prompt_cnt = 0;
    int gen_token_cnt = 0;

    int max_running_batch = 0;
    int pending_task_size = 0;

    double prepare_duration = 0;
    double model_duration = 0;
    double sampling_duration = 0;
    double total_duration = 0;
    double set_input_duration = 0;
    double send_duration = 0;
    double early_finish_duration = 0;

    double step_prepare_duration = 0;
    double step_set_input_duration = 0;
    double step_model_duration = 0;
    double step_sampling_duration = 0;
    double step_send_duration = 0;
    double step_early_finish_duration = 0;
    double step_total_duration = 0;
};

class DecodeAndSendTask final {
public:
    // range is [start_id, end_id)
    DecodeAndSendTask(uint32_t start_id, uint32_t end_id, const utils::Tokenizer* tokenizer,
                      std::vector<TidGenToken>* tid_gen_token_list, Connection* c)
        : start_id_(start_id)
        , end_id_(end_id)
        , tokenizer_(tokenizer)
        , tid_gen_token_list_(tid_gen_token_list)
        , conn_(c) {}

    RetCode Process() {
        // for circumstance task_num < thread_num
        if (start_id_ >= tid_gen_token_list_->size()) {
            return RC_SUCCESS;
        }

        vector<Response> rsp_list(end_id_ - start_id_);
        for (uint32_t i = start_id_; i < end_id_; ++i) {
            const auto& tid_gen_token = tid_gen_token_list_->at(i);
            int token = tid_gen_token.token;
            Response& rsp = rsp_list[i - start_id_];
            if (tid_gen_token.is_token_in_out) {
                rsp.token = token;
            } else {
                tokenizer_->Decode(&token, 1, &rsp.generated);
            }
            rsp.id = tid_gen_token.tid;
            rsp.flag = tid_gen_token.is_last ? Response::IS_LAST : Response::NORMAL;
        }
        conn_->Send(rsp_list);

        return RC_SUCCESS;
    }

private:
    uint32_t start_id_;
    uint32_t end_id_;
    const utils::Tokenizer* tokenizer_;
    std::vector<TidGenToken>* tid_gen_token_list_;
    Connection* conn_;
};

#ifdef PPL_LLM_ENABLE_PROFILING
static void PrintProfilingMsg(const Profiler& profiler, int step, int running_batch, uint64_t kv_max_blk,
                              uint64_t kv_rest_blk) {
    fprintf(stderr, "[PERF] --- step %d -------------------------------------------------\n", step);
    fprintf(stderr, "[PERF]  |- ");
    ::PrintMemUsage();
    fprintf(stderr, "[PERF]  |- kv cache usage: %.2f %%\n", (1.0f - (double)kv_rest_blk / kv_max_blk) * 100.0);
    fprintf(stderr, "[PERF]  |- pending task number: %d\n", profiler.pending_task_size);
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
    fprintf(stderr, "[PERF]  |-- send response    | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_send_duration, profiler.send_duration / step, profiler.send_duration);
    fprintf(stderr, "[PERF]  |-- early finish     | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_early_finish_duration, profiler.early_finish_duration / step, profiler.early_finish_duration);

    fprintf(stderr, "[PERF]  |- schedule cost: %.2f %%\n",
            (profiler.total_duration - profiler.model_duration) / profiler.total_duration * 100);
}
#endif

#ifdef PPL_LLM_ENABLE_DEBUG
#include <unistd.h>
template <class T>
static void PrintVector(vector<T> vec) {
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

static bool SaveOutputsOneByOne(const Runtime* runtime, const string& tag = "") {
    string g_flag_save_data_dir = ".";
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

static bool SaveInputsOneByOne(const Runtime* runtime, const string& tag = "") {
    string g_flag_save_data_dir = ".";

    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto shape = t->GetShape();

        auto bytes = shape->CalcBytesIncludingPadding();
        vector<char> buffer(bytes);

        ppl::nn::TensorShape src_desc = *t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        if (string(t->GetName()) == "kv_cache" || string(t->GetName()) == "kv_scale") {
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

static void PrintInputInfo(const Runtime* runtime) {
    LOG(INFO) << "----- input info -----";
    for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
        auto tensor = runtime->GetInputTensor(i);
        LOG(INFO) << "input[" << i << "]:";
        LOG(INFO) << "    name: " << tensor->GetName();

        string dims_str;
        auto shape = tensor->GetShape();
        for (uint32_t j = 0; j < shape->GetDimCount(); ++j) {
            dims_str += " " + ToString(shape->GetDim(j));
        }
        LOG(INFO) << "    dim(s):" << dims_str;

        LOG(INFO) << "    data type: " << GetDataTypeStr(shape->GetDataType());
        LOG(INFO) << "    data format: " << GetDataFormatStr(shape->GetDataFormat());
        LOG(INFO) << "    byte(s) excluding padding: " << shape->CalcBytesExcludingPadding();
        LOG(INFO) << "    buffer address: " << tensor->GetBufferPtr();

        const int64_t elem_count = tensor->GetShape()->CalcElementsExcludingPadding();
        if (tensor->GetShape()->GetDataType() == ppl::common::DATATYPE_INT64 && elem_count <= 16) {
            std::vector<int64_t> vals(elem_count, 0);
            if (ppl::common::RC_SUCCESS != tensor->CopyToHost(vals.data())) {
                LOG(ERROR) << "[" << tensor->GetName() << "] CopyToHost FAILED";
            } else {
                std::string val_str = "";
                for (uint32_t j = 0; j < elem_count; ++j) {
                    val_str += std::to_string(vals[j]) + " ";
                }
                LOG(INFO) << "    value(s): " << val_str;
            }
        }
    }

    LOG(INFO) << "----------------------";
}
#endif

RetCode LLaMAWorker::CheckParameters() const {
    if (model_config_.auto_causal != true) {
        LOG(ERROR) << "only support auto_causal == true";
        return RC_INVALID_VALUE;
    }

    if (model_config_.cache_mode != 0) {
        LOG(ERROR) << "only support cache_mode == 0";
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

LLaMAWorker::LLaMAWorker(const Resource& resource, const ModelConfig& mconfig, const WorkerConfig& wconfig,
                         Connection* c)
    : RequestProcessor(c)
    , tokenizer_(resource.tokenizer)
    , model_config_(mconfig)
    , worker_config_(wconfig)
    , device_worker_pool_(resource.device_worker_pool)
    , tensor_parallel_size_(resource.tensor_parallel_size)
    , worker_thread_args_(resource.tensor_parallel_size)
    , sampler_(resource.sampler) {
    pthread_mutex_init(&tid_shutdown_lock_, nullptr);

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
        if (model_config_.cache_quant_bit > 0) {
            arg->kv_scale = arg->resource->runtime->GetInputTensor(10);
        }

        arg->logits = arg->resource->runtime->GetOutputTensor(0);

        arg->decoding_batches->SetDeviceContext(arg->resource->host_device);
        arg->max_seq_len->SetDeviceContext(arg->resource->host_device);
        arg->max_kv_len->SetDeviceContext(arg->resource->host_device);

        arg->kv_cache->SetBufferPtr(arg->resource->kv_cache_mem);
        if (model_config_.cache_quant_bit > 0) {
            arg->kv_scale->SetBufferPtr(arg->resource->kv_scale_mem);
        }
    }
}

LLaMAWorker::~LLaMAWorker() {
    if (worker_thread_active_) {
        pthread_mutex_lock(sched_.GetQueueLock()); // to ensure that workers are waiting for status changing
        worker_thread_active_ = false;
        pthread_cond_signal(&req_signal_);
        pthread_mutex_unlock(sched_.GetQueueLock());
        pthread_join(worker_thread_, nullptr);
    }

    pthread_cond_destroy(&req_signal_);
    pthread_mutex_destroy(&tid_shutdown_lock_);
}

RetCode LLaMAWorker::Init() {
    auto ret = CheckParameters();
    if (ret != RC_SUCCESS) {
        LOG(ERROR) << "CheckParameters failed.";
        return ret;
    }

    for (int i = 0; i < tensor_parallel_size_; i++) {
        auto arg = &worker_thread_args_[i];
        if (model_config_.cache_layout == 0) {
            arg->kv_cache->GetShape()->Reshape({(int64_t)kv_cache_max_tokens_, model_config_.num_layers, 2,
                                                model_config_.num_kv_heads / tensor_parallel_size_,
                                                model_config_.hidden_dim / model_config_.num_heads});
            if (model_config_.cache_quant_bit > 0) {
                arg->kv_scale->GetShape()->Reshape(
                    {(int64_t)kv_cache_max_tokens_, model_config_.num_layers, 2,
                     model_config_.num_kv_heads / tensor_parallel_size_,
                     model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});
            }

        } else if (model_config_.cache_layout == 1) {
            arg->kv_cache->GetShape()->Reshape({model_config_.num_layers, (int64_t)kv_cache_max_tokens_, 2,
                                                model_config_.num_kv_heads / tensor_parallel_size_,
                                                model_config_.hidden_dim / model_config_.num_heads});
            if (model_config_.cache_quant_bit > 0) {
                arg->kv_scale->GetShape()->Reshape(
                    {model_config_.num_layers, (int64_t)kv_cache_max_tokens_, 2,
                     model_config_.num_kv_heads / tensor_parallel_size_,
                     model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});
            }
        } else if (model_config_.cache_layout == 2) {
            arg->kv_cache->GetShape()->Reshape({model_config_.num_layers, 2, (int64_t)kv_cache_max_tokens_,
                                                model_config_.num_kv_heads / tensor_parallel_size_,
                                                model_config_.hidden_dim / model_config_.num_heads});
            if (model_config_.cache_quant_bit > 0) {
                arg->kv_scale->GetShape()->Reshape(
                    {model_config_.num_layers, 2, (int64_t)kv_cache_max_tokens_,
                     model_config_.num_kv_heads / tensor_parallel_size_,
                     model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});
            }
        } else if (model_config_.cache_layout == 3) {
            arg->kv_cache->GetShape()->Reshape(
                {model_config_.num_layers, 2, model_config_.num_kv_heads / tensor_parallel_size_,
                 (int64_t)kv_cache_max_tokens_, model_config_.hidden_dim / model_config_.num_heads});
            if (model_config_.cache_quant_bit > 0) {
                arg->kv_scale->GetShape()->Reshape(
                    {model_config_.num_layers, 2, model_config_.num_kv_heads / tensor_parallel_size_,
                     (int64_t)kv_cache_max_tokens_,
                     model_config_.hidden_dim / model_config_.num_heads / model_config_.cache_quant_group});
            }
        } else {
            LOG(ERROR) << "impossible status: cache_layout = [" << model_config_.cache_layout << "]";
        }
    }

    ret = decoder_thread_pool_.Init(DECODER_THREAD_NUM);
    if (ret != RC_SUCCESS) {
        LOG(ERROR) << "Init decoder thread pool error";
        return RC_OTHER_ERROR;
    }

    worker_thread_active_ = true;
    pthread_cond_init(&req_signal_, nullptr);
    auto err = pthread_create(&worker_thread_, nullptr, WorkerThreadFunc, this);
    if (err != 0) {
        worker_thread_active_ = false;
        LOG(ERROR) << "create worker thread failed.";
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

struct RequestCheckResult final {
    int64_t cache_index;
    int rest_iters;
    int first_fill_len;
    int max_tokens_per_step;
};

static bool ParseRequest(const LlamaRequest& req, const RequestCheckResult& check_res,
                         WorkerController* worker_controller, unordered_map<uint64_t, TidController>* tid_controllers,
                         Connection* conn) {
    if (check_res.rest_iters < 0) {
        conn->NotifyFailure(req.orig->id);
        return true;
    }

    if (check_res.cache_index == INT64_MAX) {
        LOG(ERROR) << "catch invalid cache_index.";
        return false;
    }

    auto tid = req.orig->id;
    auto& tid_ctrl = tid_controllers->emplace(tid, TidController()).first->second;
    tid_ctrl.tid = tid;
    tid_ctrl.temperature = req.orig->temperature;
    tid_ctrl.early_stopping = req.orig->early_stopping;
    tid_ctrl.rest_iters = check_res.rest_iters;
    tid_ctrl.first_fill_len = check_res.first_fill_len;
    tid_ctrl.total_len = check_res.first_fill_len + check_res.rest_iters;
    tid_ctrl.cache_index = check_res.cache_index;
    tid_ctrl.next_tokens = req.token_id_list;
    tid_ctrl.stop_tokens = req.stop_tokens;
    tid_ctrl.is_token_in_out = req.is_token_in_out;

    worker_controller->tid_list.push_back(&tid_ctrl);
    worker_controller->start_pos.push_back(0);
    worker_controller->temperatures.push_back(tid_ctrl.temperature);
    worker_controller->cache_indices.push_back(check_res.cache_index);
    worker_controller->total_rest_iters = std::max<int64_t>(worker_controller->total_rest_iters, tid_ctrl.rest_iters);

    return true;
}

void LLaMAWorker::ClearTask(uint64_t tid) {
    pthread_mutex_lock(&tid_shutdown_lock_);
    worker_controller_.tid_shutdown.insert(tid);
    pthread_mutex_unlock(&tid_shutdown_lock_);
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

void LLaMAWorker::DeleteTask(const vector<uint64_t>& finished_list,
                             unordered_map<uint64_t, TidController>* tid_controllers) {
    // process finished task
    for (size_t i = 0; i < finished_list.size(); ++i) {
        auto tid = finished_list[i];
        auto tid_it = tid_controllers->find(tid);
        if (tid_it == tid_controllers->end()) {
            LOG(ERROR) << "find non exist tid: " << tid;
            continue;
        }
        auto& tid_ctrl = tid_it->second;

        if (tid_ctrl.is_first_fill == false) { // avoid corner case generation_len = 1
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

        idx_mgr_.Free(tid_ctrl.cache_index, tid_ctrl.total_len - 1);
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

static void UpdateInput(const unordered_map<uint64_t, TidController>& tid_controllers,
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

        worker_controller->token_inputs.insert(worker_controller->token_inputs.end(), tid_ctrl->next_tokens->begin(),
                                               tid_ctrl->next_tokens->end());
        worker_controller->seq_starts.push_back(worker_controller->seq_starts[i] + tid_ctrl->next_tokens->size());
        worker_controller->kv_starts.push_back(worker_controller->kv_starts[i] + tid_ctrl->first_fill_len +
                                               tid_ctrl->passed_iters);

        worker_controller->max_seq_len =
            std::max<int64_t>(tid_ctrl->next_tokens->size(), worker_controller->max_seq_len);
        worker_controller->max_kv_len =
            std::max<int64_t>(tid_ctrl->first_fill_len + tid_ctrl->passed_iters, worker_controller->max_kv_len);
    }
}

class SetInputTask final {
public:
    SetInputTask(uint32_t id, const WorkerController* wc, WorkerThreadArg* arg_list)
        : id_(id), wc_(wc), arg_list_(arg_list) {}

    RetCode Process() {
        arg_list_[id_].token_ids->FreeBuffer();
        arg_list_[id_].seq_starts->FreeBuffer();
        arg_list_[id_].kv_starts->FreeBuffer();
        arg_list_[id_].cache_indices->FreeBuffer();
        arg_list_[id_].start_pos->FreeBuffer();
        arg_list_[id_].logits->FreeBuffer();

        RetCode rc;
        // token ids
        arg_list_[id_].token_ids->GetShape()->Reshape({int64_t(wc_->token_inputs.size())});
        rc = arg_list_[id_].token_ids->CopyFromHostAsync(wc_->token_inputs.data());
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set token_ids [" << arg_list_[id_].token_ids->GetName() << "] failed: " << GetRetCodeStr(rc);
            return rc;
        }

        // seq_start
        arg_list_[id_].seq_starts->GetShape()->Reshape({int64_t(wc_->seq_starts.size())});
        rc = arg_list_[id_].seq_starts->CopyFromHostAsync(wc_->seq_starts.data());
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set seq_starts [" << arg_list_[id_].seq_starts->GetName()
                       << "] failed: " << GetRetCodeStr(rc);
            return rc;
        }

        // kv_starts
        arg_list_[id_].kv_starts->GetShape()->Reshape({int64_t(wc_->kv_starts.size())});
        rc = arg_list_[id_].kv_starts->CopyFromHostAsync(wc_->kv_starts.data());
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set kv_starts " << arg_list_[id_].kv_starts->GetName() << " failed: " << GetRetCodeStr(rc);
            return rc;
        }

        // cache_indices
        arg_list_[id_].cache_indices->GetShape()->Reshape({int64_t(wc_->cache_indices.size())});
        rc = arg_list_[id_].cache_indices->CopyFromHostAsync(wc_->cache_indices.data());
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set cache_indices [" << arg_list_[id_].cache_indices->GetName()
                       << "] failed: " << GetRetCodeStr(rc);
            return rc;
        }

        // decoding batches
        rc = arg_list_[id_].decoding_batches->CopyFromHostAsync(&wc_->decoding_batches);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set decoding_batches [" << arg_list_[id_].decoding_batches->GetName()
                       << "] failed: " << GetRetCodeStr(rc);
            return rc;
        }

        // start_pos
        arg_list_[id_].start_pos->GetShape()->Reshape({int64_t(wc_->start_pos.size())});
        rc = arg_list_[id_].start_pos->CopyFromHostAsync(wc_->start_pos.data());
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set start_pos [" << arg_list_[id_].start_pos->GetName() << "] failed: " << GetRetCodeStr(rc);
            return rc;
        }

        // max_seq_len
        rc = arg_list_[id_].max_seq_len->CopyFromHostAsync(&wc_->max_seq_len);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set max_seq_len [" << arg_list_[id_].max_seq_len->GetName()
                       << "] failed: " << GetRetCodeStr(rc);
            return rc;
        }

        // max_kv_len
        rc = arg_list_[id_].max_kv_len->CopyFromHostAsync(&wc_->max_kv_len);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set max_kv_len [" << arg_list_[id_].max_kv_len->GetName()
                       << "] failed: " << GetRetCodeStr(rc);
            return rc;
        }

        rc = arg_list_[id_].resource->runtime->Synchronize();
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "set input tensor synchronize fail";
            return rc;
        }
        return rc;
    }

private:
    const uint32_t id_;
    const WorkerController* wc_;
    WorkerThreadArg* arg_list_;
};

class RunModelTask final {
public:
    RunModelTask(uint32_t id, WorkerThreadArg* arg_list) : id_(id), arg_list_(arg_list) {}

    RetCode Process() {
        return arg_list_[id_].resource->runtime->Run();
    }

private:
    const uint32_t id_;
    WorkerThreadArg* arg_list_;
};

void LLaMAWorker::Work() {
    RetCode rc;

    Profiler profiler;
    worker_controller_.Reset();

    unordered_map<uint64_t, TidController> tid_controllers;
    std::vector<TidGenToken> tid_gen_token_list;

    int running_batch = 0;
    long long step = 0;
    int cache_cool_down_count = 0;
    RequestCheckResult check_res;
    auto check_func = [this, &check_res, &cache_cool_down_count](const LlamaRequest& req) -> bool {
        check_res.cache_index = INT64_MAX;
        check_res.rest_iters = -1;
        check_res.first_fill_len = req.token_id_list->size();
        if (check_res.max_tokens_per_step > worker_config_.max_tokens_per_step) {
            return false;
        }
        check_res.max_tokens_per_step += check_res.first_fill_len;

        if (check_res.first_fill_len + req.orig->generation_length > (size_t)worker_config_.max_tokens_per_request) {
            check_res.rest_iters = worker_config_.max_tokens_per_request - check_res.first_fill_len;
        } else {
            check_res.rest_iters = req.orig->generation_length;
        }

        if (check_res.rest_iters < 0) {
            return true;
        }

        check_res.cache_index = idx_mgr_.Alloc(check_res.first_fill_len + check_res.rest_iters - 1);

        if (check_res.cache_index == INT64_MAX) {
            cache_cool_down_count = std::min(std::max(1, (int)floorf(worker_controller_.tid_list.size() * 0.1f)), 8);
            return false;
        }

        return true;
    };

    decoder_thread_pool_.RunAsync([](uint32_t, uint32_t) {});

    while (true) {
        // Recv and Parse Requset
        auto global_start = std::chrono::high_resolution_clock::now();

        check_res.max_tokens_per_step = running_batch;
        {
            utils::TimingGuard __timing__(&profiler.step_prepare_duration);
            if (worker_controller_.tid_list.size() < (size_t)worker_config_.max_running_batch &&
                cache_cool_down_count <= 0) {
                while (true) {
                    auto req = sched_.TryPopRequest(check_func);
                    if (!req) {
                        break;
                    }

                    if (!ParseRequest(*req, check_res, &worker_controller_, &tid_controllers, conn_)) {
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

            running_batch = worker_controller_.tid_list.size();
            if (running_batch == 0) {
                break;
            }

            UpdateInput(tid_controllers, &worker_controller_);
            profiler.max_running_batch = std::max(running_batch, profiler.max_running_batch);
        }
        profiler.prepare_duration += profiler.step_prepare_duration;

        LOG(DEBUG) << "Step: " << step << " ---------------------------------";

        // set inputs tensor
        {
            utils::TimingGuard __timing__(&profiler.step_set_input_duration);
            rc = utils::ParallelExecute<SetInputTask>(device_worker_pool_, &worker_controller_,
                                                      worker_thread_args_.data());
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "ParallelExecute(SetInputTask) failed.";
                break;
            }
        }
        profiler.set_input_duration += profiler.step_set_input_duration;

        // model forward
        {
            utils::TimingGuard __timing__(&profiler.step_model_duration);
            rc = utils::ParallelExecute<RunModelTask>(device_worker_pool_, worker_thread_args_.data());
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "ParallelExecute(RunModelTask) failed: " << GetRetCodeStr(rc);
                break;
            }
        }
        profiler.model_duration += profiler.step_model_duration;

        // sampling
        vector<int32_t> gen_tokens(running_batch);
        {
            utils::TimingGuard __timing__(&profiler.step_sampling_duration);

            auto logits = worker_thread_args_[0].logits;
            rc = sampler_->SampleTopPTopK((float*)logits->GetBufferPtr(), worker_controller_.temperatures.data(),
                                          running_batch, model_config_.vocab_size, worker_config_.top_p,
                                          worker_config_.top_k, gen_tokens.data());
            if (rc != RC_SUCCESS) {
                LOG(ERROR) << "SampleTopPTopK failed: " << GetRetCodeStr(rc);
                break;
            }
        }
        profiler.sampling_duration += profiler.step_sampling_duration;
        profiler.gen_token_cnt += running_batch;

        // send stream chat rsp
        {
            utils::TimingGuard __timing__(&profiler.step_send_duration);
            decoder_thread_pool_.Wait();

            tid_gen_token_list.clear();
            for (int task_iter = 0; task_iter < running_batch; ++task_iter) {
                auto* tid_ctrl = worker_controller_.tid_list[task_iter];
                int gen_token = gen_tokens[task_iter];
                LOG(DEBUG) << "task[" << tid_ctrl->tid << "] gen token: " << gen_token;
                tid_ctrl->next_tokens.reset(new vector<int>({gen_token}));
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

                tid_gen_token_list.emplace_back(tid_ctrl->tid, gen_token, false, tid_ctrl->is_token_in_out);
                // finished task
                bool is_shutdown =
                    worker_controller_.tid_shutdown.find(tid_ctrl->tid) != worker_controller_.tid_shutdown.end();
                if (tid_ctrl->rest_iters <= 0 || (tid_ctrl->early_stopping && tid_ctrl->stop_tokens->find(gen_token) != tid_ctrl->stop_tokens->end()) || is_shutdown) {
                    tid_gen_token_list.back().is_last = true;
                    if (cache_cool_down_count > 0)
                        cache_cool_down_count--;
                    worker_controller_.tid_finished.push_back(tid_ctrl->tid);
                }
            }
            worker_controller_.tid_shutdown.clear();

            decoder_thread_pool_.RunAsync([this, &tid_gen_token_list](uint32_t nthr, uint32_t ithr) {
                size_t task_size = tid_gen_token_list.size();
                uint32_t n_block = task_size % nthr == 0 ? task_size / nthr : task_size / nthr + 1;
                uint32_t start_id = ithr * n_block;
                uint32_t end_id = (ithr + 1) * n_block > task_size ? task_size : (ithr + 1) * n_block;

                auto task = DecodeAndSendTask(start_id, end_id, tokenizer_, &tid_gen_token_list, conn_);
                task.Process();
            });
        }
        profiler.send_duration += profiler.step_send_duration;

        {
            utils::TimingGuard __timing__(&profiler.step_early_finish_duration);
            // update iteration
            step++;
            worker_controller_.total_rest_iters--;
            LOG(DEBUG) << "Rest iters: " << worker_controller_.total_rest_iters;

            // early finish
            if (worker_controller_.tid_finished.size() > 0) {
                LOG(DEBUG) << "Do early finish";
                profiler.prompt_cnt += worker_controller_.tid_finished.size();
                DeleteTask(worker_controller_.tid_finished, &tid_controllers);
            }
        }
        profiler.early_finish_duration += profiler.step_early_finish_duration;

        auto global_end = std::chrono::high_resolution_clock::now();
        profiler.step_total_duration =
            double(std::chrono::duration_cast<std::chrono::microseconds>(global_end - global_start).count()) / 1000.0;
        profiler.total_duration += profiler.step_total_duration;
        profiler.pending_task_size = sched_.GetPendingSize();

#ifdef PPL_LLM_ENABLE_PROFILING
        if (step % 100 == 0 || worker_controller_.total_rest_iters == 0) {
            PrintProfilingMsg(profiler, step, running_batch, kv_cache_max_tokens_, idx_mgr_.GetAvailableBlockNum());
        }
#endif
    }

    decoder_thread_pool_.Wait();
}

void LLaMAWorker::Process(const shared_ptr<Request>& req) {
    auto lreq = make_shared<LlamaRequest>();
    lreq->orig = req;
    if (!req->token_ids) {
        lreq->token_id_list = std::make_shared<std::vector<int>>();
        tokenizer_->Encode(req->prompt.data(), req->prompt.size(), lreq->token_id_list.get());
        lreq->stop_tokens = std::make_shared<std::unordered_set<int>>();
        lreq->stop_tokens->insert(tokenizer_->GetEosId());
        lreq->is_token_in_out = false;
        conn_->OnTokenize(req->id, *lreq->token_id_list);
    } else {
        lreq->token_id_list = req->token_ids;
        lreq->stop_tokens = req->stop_tokens;
        lreq->is_token_in_out = true;
    }

    sched_.PushRequest(lreq);
    if (sched_.GetPendingSize() == 1) {
        pthread_cond_signal(&req_signal_);
    }
}

void* LLaMAWorker::WorkerThreadFunc(void* arg) {
    auto worker = (LLaMAWorker*)arg;
    while (true) {
        pthread_mutex_lock(worker->sched_.GetQueueLock());
        while (worker->sched_.GetPendingSize() == 0) {
            if (!worker->worker_thread_active_) {
                pthread_mutex_unlock(worker->sched_.GetQueueLock());
                return nullptr;
            }
            LOG(INFO) << "waiting for request ...";
            pthread_cond_wait(&worker->req_signal_, worker->sched_.GetQueueLock());
        }
        pthread_mutex_unlock(worker->sched_.GetQueueLock());
        worker->Work();
    }
    return nullptr;
}

}}} // namespace ppl::llm::llama
