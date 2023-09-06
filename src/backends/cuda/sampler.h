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

#ifndef __SERVING_SAMPLER_CUDA_SAMPLER_H__
#define __SERVING_SAMPLER_CUDA_SAMPLER_H__

#include "utils/sampler.h"
#include <cuda_runtime.h>

namespace ppl { namespace llm { namespace cuda {

class TopPTopKSampler final : public utils::Sampler {
public:
    TopPTopKSampler(cudaStream_t stream, int32_t vocab_size, float top_p, float top_k)
        : vocab_size_(vocab_size), top_p_(top_p), top_k_(top_k), stream_(stream) {}
    virtual ~TopPTopKSampler() {
        Clear();
    }

    ppl::common::RetCode Init();
    ppl::common::RetCode Process(const float* logits_device, const float* temperatures_host, int32_t batch,
                                 int32_t* output_host) override;

private:
    void Clear();

private:
    const int32_t vocab_size_;
    const float top_p_;
    const float top_k_;
    cudaStream_t stream_ = 0;
    int32_t* cu_output_ = nullptr;
    int64_t cu_output_size_ = 0;
};

}}}; // namespace ppl::llm::cuda

#endif
