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

#include "post_processor.h"

#include "ppl/nn/engines/llm_cuda/options.h"
#include "ppl/common/log.h"
#include "ppl/kernel/llm/cuda/pmx/sample.h"
#include "ppl/kernel/llm/cuda/pmx/penalty.h"

using namespace ppl::common;

namespace ppl { namespace llm { namespace cuda {

void CudaPostProcessor::Clear() {
    auto* sampler_buffer = temperatures_device_;
    if (sampler_buffer) {
        auto err = cudaFreeAsync(sampler_buffer, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaFreeAsync failed: " << cudaGetErrorString(err);
            return;
        }
        temperatures_device_ = nullptr;
        top_k_device_ = nullptr;
        top_p_device_ = nullptr;
        output_device_ = nullptr;
        logprobs_device_ = nullptr;
    }
    if (workspace_) {
        auto err = cudaFreeAsync(workspace_, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaFreeAsync failed: " << cudaGetErrorString(err);
            return;
        }
        workspace_ = nullptr;
    }
    auto* penalty_buffer = penalty_count_map_;
    if (penalty_buffer) {
        auto err = cudaFreeAsync(penalty_buffer, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaFreeAsync failed: " << cudaGetErrorString(err);
            return;
        }
        penalty_count_map_ = nullptr;
        batch_slots_device_ = nullptr;
        repetition_penalties_device_ = nullptr;
        presence_penalties_device_ = nullptr;
        frequency_penalties_device_ = nullptr;
    }
    auto err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);     
        return;
    }
}

RetCode CudaPostProcessor::InitPostProcessorMem(int max_running_batch, int vocab_size, bool enable_penalty) {
    uint32_t temperature_bytes = max_running_batch * sizeof(float);
    uint32_t top_k_bytes = max_running_batch * sizeof(int32_t);
    uint32_t top_p_bytes = max_running_batch * sizeof(float);
    uint32_t rand_bytes = max_running_batch * sizeof(float);
    uint32_t output_bytes = max_running_batch * sizeof(int32_t);
    uint32_t logprob_bytes = max_running_batch * sizeof(float);
    uint32_t needed_mem_bytes = temperature_bytes + top_k_bytes + top_p_bytes + rand_bytes + output_bytes + logprob_bytes;

    char* sampler_buffer;
    cudaError_t cu_ret = cudaMalloc((void**)&sampler_buffer, needed_mem_bytes);
    if (cu_ret != cudaSuccess) {
        LOG(ERROR) << "cudaMallocAsync: [" << needed_mem_bytes << "] failed: " << cudaGetErrorString(cu_ret);
        return RC_OTHER_ERROR;
    }

    temperatures_device_ = (float*)sampler_buffer;
    top_k_device_ = (int32_t*)(sampler_buffer + temperature_bytes);
    top_p_device_ = (float*)(sampler_buffer + temperature_bytes + top_k_bytes);
    rand_device_ = (float*)(sampler_buffer + temperature_bytes + top_k_bytes + top_p_bytes);
    output_device_ = (int32_t*)(sampler_buffer + temperature_bytes + top_k_bytes + top_p_bytes + rand_bytes);
    logprobs_device_ = (float*)(sampler_buffer + temperature_bytes + top_k_bytes + top_p_bytes + rand_bytes + output_bytes);

    if (enable_penalty) {
        uint32_t penalty_map_bytes = max_running_batch * vocab_size * sizeof(uint16_t);
        uint32_t batch_slots_bytes = max_running_batch * sizeof(int64_t);
        uint32_t repetition_penalties_bytes = max_running_batch * sizeof(float);
        uint32_t presence_penalties_bytes = max_running_batch * sizeof(float);
        uint32_t frequency_penalties_bytes = max_running_batch * sizeof(float);
        uint32_t needed_mem_bytes = penalty_map_bytes + batch_slots_bytes + repetition_penalties_bytes +
            presence_penalties_bytes + frequency_penalties_bytes;

        char* penalty_buffer;
        cudaError_t cu_ret = cudaMalloc((void**)&penalty_buffer, needed_mem_bytes);
        if (cu_ret != cudaSuccess) {
            LOG(ERROR) << "cudaMallocAsync: [" << needed_mem_bytes << "] failed: " << cudaGetErrorString(cu_ret);
            return RC_OTHER_ERROR;
        }

        penalty_count_map_ = (uint16_t*)penalty_buffer;
        batch_slots_device_ = (int64_t*)(penalty_buffer + penalty_map_bytes);
        repetition_penalties_device_ = (float*)(penalty_buffer + penalty_map_bytes + batch_slots_bytes);
        presence_penalties_device_ =
            (float*)(penalty_buffer + penalty_map_bytes + batch_slots_bytes + repetition_penalties_bytes);
        frequency_penalties_device_ = (float*)(penalty_buffer + penalty_map_bytes + batch_slots_bytes +
                                               repetition_penalties_bytes + presence_penalties_bytes);
    }
    return RC_SUCCESS;
}

RetCode CudaPostProcessor::SampleTopKTopP(const float* logits_device, const float* temperatures,
                                      const int32_t* top_k_host, const float* top_p_host, int32_t batch,
                                      int32_t vocab_size, int32_t batch_stride, int32_t default_top_k,
                                      float default_top_p, bool req_list_changed, int32_t* output_host,
                                      float* logprobs_host, bool enable_penalty) {
    const float* temperatures_host = enable_penalty ? nullptr : temperatures;
    const int64_t output_size = batch * sizeof(int32_t);
    const int64_t logprob_size = batch * sizeof(float);
    const int64_t temperatures_size = temperatures_host ? batch * sizeof(float) : 0;
    const int64_t top_p_size = top_p_host ? batch * sizeof(float) : 0;
    cudaError_t err;
    int64_t needed_mem_size = 0;
    if (default_top_k > 0) {
        needed_mem_size =
            ppl::kernel::llm::cuda::pmx::sample_topk_topp_get_workspace_size(batch, vocab_size, default_top_k);
    }
    if (needed_mem_size > workspace_size_) {
        if (workspace_) {
            err = cudaFreeAsync(workspace_, stream_);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaFreeAsync failed: " << cudaGetErrorString(err);
                return RC_DEVICE_MEMORY_ERROR;
            }
        }
        err = cudaMallocAsync(&workspace_, needed_mem_size, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMallocAsync failed: " << cudaGetErrorString(err);
            return RC_OUT_OF_MEMORY;
        }
        workspace_size_ = needed_mem_size;
    }

    RetCode rc;
    float* temperatures_optional = nullptr;
    float* top_p_optional = nullptr;

    if (req_list_changed) {
        if (temperatures_host) {
            err = cudaMemcpyAsync(temperatures_device_, temperatures_host, temperatures_size, cudaMemcpyHostToDevice,
                                  stream_);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaMemcpyAsync temperatures failed: " << cudaGetErrorString(err);
                return RC_DEVICE_MEMORY_ERROR;
            }
            temperatures_optional = temperatures_device_;
        }

        if (top_p_host) {
            err = cudaMemcpyAsync(top_p_device_, top_p_host, top_p_size, cudaMemcpyHostToDevice, stream_);

            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaMemcpyAsync top p failed: " << cudaGetErrorString(err);
                return RC_DEVICE_MEMORY_ERROR;
            }
            top_p_optional = top_p_device_;
        }
    }

    const float default_rand_val = static_cast<float>(rand()) / RAND_MAX;
    std::vector<float> rand_host(batch);
    for (auto& rnd_val : rand_host) {
        rnd_val = static_cast<float>(rand()) / RAND_MAX;
    }
    err = cudaMemcpyAsync(rand_device_, rand_host.data(), rand_host.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync rand val failed: " << cudaGetErrorString(err);
        return RC_DEVICE_MEMORY_ERROR;
    }

    rc = ppl::kernel::llm::cuda::pmx::sample_topk_topp(stream_, logits_device, temperatures_optional, top_p_optional,
                                                       rand_device_, batch, vocab_size, batch_stride, default_top_k,
                                                       default_top_p, default_rand_val, workspace_, output_device_,
                                                       logprobs_device_);

    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "sampling kernel failed: " << GetRetCodeStr(rc);
        return rc;
    }

    err = cudaMemcpyAsync(output_host, output_device_, output_size, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync output failed: " << cudaGetErrorString(err);
        return RC_DEVICE_MEMORY_ERROR;
    }

    err = cudaMemcpyAsync(logprobs_host, logprobs_device_, logprob_size, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync logprobs failed: " << cudaGetErrorString(err);
        return RC_DEVICE_MEMORY_ERROR;
    }

    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);
        return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
}

RetCode CudaPostProcessor::ApplyPenalty(const float* temperatures_host, const float* repetition_penalties_host,
                                    const float* presence_penalties_host, const float* frequency_penalties_host,
                                    const int64_t* batch_slots_host, const int64_t* token_inputs,
                                    const int64_t* seqstarts, const int64_t* start_pos, int32_t batch,
                                    int32_t vocab_size, bool req_list_changed, float* logits) {
    cudaError_t err;
    float* presence_penalties_optional = nullptr;
    float* frequency_penalties_optional = nullptr;

    if (req_list_changed) {
        err = cudaMemcpyAsync(temperatures_device_, temperatures_host, batch * sizeof(float), cudaMemcpyHostToDevice,
                              stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMemcpyAsync temperature failed: " << cudaGetErrorString(err);
            return RC_DEVICE_MEMORY_ERROR;
        }
        err = cudaMemcpyAsync(batch_slots_device_, batch_slots_host, batch * sizeof(int64_t), cudaMemcpyHostToDevice,
                              stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMemcpyAsync batch_slots failed: " << cudaGetErrorString(err);
            return RC_DEVICE_MEMORY_ERROR;
        }

        err = cudaMemcpyAsync(repetition_penalties_device_, repetition_penalties_host, batch * sizeof(float),
                              cudaMemcpyHostToDevice, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMemcpyAsync repetition_penalties failed: " << cudaGetErrorString(err);
            return RC_DEVICE_MEMORY_ERROR;
        }

        if (presence_penalties_host) {
            err = cudaMemcpyAsync(presence_penalties_device_, presence_penalties_host, batch * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaMemcpyAsync presence_penalties failed: " << cudaGetErrorString(err);
                return RC_DEVICE_MEMORY_ERROR;
            }
            presence_penalties_optional = presence_penalties_device_;
        }

        if (frequency_penalties_host) {
            err = cudaMemcpyAsync(frequency_penalties_device_, frequency_penalties_host, batch * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaMemcpyAsync frequency_penalties failed: " << cudaGetErrorString(err);
                return RC_DEVICE_MEMORY_ERROR;
            }
            frequency_penalties_optional = frequency_penalties_device_;
        }
    }
    auto rc = ppl::kernel::llm::cuda::pmx::apply_penalty(
        stream_, logits, temperatures_device_, repetition_penalties_device_, presence_penalties_optional,
        frequency_penalties_optional, batch_slots_device_, token_inputs, seqstarts, start_pos, batch, vocab_size,
        penalty_count_map_, logits);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "apply_penalty kernel failed: " << GetRetCodeStr(rc);
        return rc;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::llm::cuda
