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

#include "sampler/sampler.h"

namespace ppl { namespace llm { namespace cuda {

class Sampler : public llm::Sampler {
public:
    Sampler(ppl::nn::DeviceContext* dev) : llm::Sampler(dev) {}
    virtual ~Sampler() {
        Clear();
    }

    ppl::common::RetCode Init();

    ppl::common::RetCode SampleTopPTopK(const float* logits_device, const float* temperatures_host, const int32_t batch,
                                        const int32_t vocab_size, const float top_p, const float top_k,
                                        int32_t* output_host) override;

private:
    void Clear();

private:
    int32_t* cu_output_ = nullptr;
    int64_t cu_output_size_ = 0;
};

}}}; // namespace ppl::llm::cuda

#endif
