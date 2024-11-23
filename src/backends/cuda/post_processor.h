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

#ifndef __PPL_LLM_CUDA_POST_PROCESSOR_H__
#define __PPL_LLM_CUDA_POST_PROCESSOR_H__

#include "../../common/post_processor.h"

#include <cuda_runtime.h>

namespace ppl { namespace llm { namespace cuda {

class CudaPostProcessor final : public PostProcessor {
public:
    CudaPostProcessor(cudaStream_t stream) : stream_(stream) {}
    virtual ~CudaPostProcessor() {
        Clear();
    }

    ppl::common::RetCode InitPostProcessorMem(int max_running_batch, int vocab_size, bool enable_penalty) override;

    ppl::common::RetCode SampleTopKTopP(const float* logits_device, const float* temperatures_host,
                                        const int32_t* top_k_host, const float* top_p_host, int32_t batch,
                                        int32_t vocab_size, int32_t batch_stride, int32_t default_top_k,
                                        float default_top_p, bool req_list_changed, int32_t* output_host,
                                        float* logprobs_host, bool enable_penalty) override;

    ppl::common::RetCode ApplyPenalty(const float* temperatures_host, const float* repetition_penalties_host,
                                      const float* presence_penalties_host, const float* frequency_penalties_host,
                                      const int64_t* batch_slots_host, const int64_t* token_inputs,
                                      const int64_t* seqstarts, const int64_t* start_pos, int32_t batch,
                                      int32_t vocab_size, bool req_list_changed, float* logits) override;

private:
    void Clear();

private:
    cudaStream_t stream_ = 0;

    int32_t* workspace_ = nullptr;
    int64_t workspace_size_ = 0;
    
    float* temperatures_device_ = nullptr;
    int32_t* top_k_device_ = nullptr;
    float* top_p_device_ = nullptr;
    float* rand_device_ = nullptr;
    int32_t* output_device_ = nullptr;
    float* logprobs_device_ = nullptr;

    uint16_t* penalty_count_map_ = nullptr;
    int64_t* batch_slots_device_ = nullptr;
    float* repetition_penalties_device_ = nullptr;
    float* presence_penalties_device_ = nullptr;
    float* frequency_penalties_device_ = nullptr;
};

}}}; // namespace ppl::llm::cuda

#endif