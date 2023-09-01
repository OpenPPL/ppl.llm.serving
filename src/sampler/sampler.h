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

#ifndef __SERVING_SAMPLER_SAMPLER_H__
#define __SERVING_SAMPLER_SAMPLER_H__

#include "ppl/common/retcode.h"
#include "ppl/nn/runtime/tensor.h"
#include "ppl/nn/common/device_context.h"

namespace ppl { namespace llm {

class Sampler {
public:
    Sampler(ppl::nn::DeviceContext* dev) : dev_(dev) {}
    virtual ~Sampler() {}

    virtual ppl::common::RetCode SampleTopPTopK(const float* logits_device, const float* temperatures_host,
                                                const int32_t batch, const int32_t vocab_size, const float top_p,
                                                const float top_k, int32_t* output_host) = 0;

protected:
    ppl::nn::DeviceContext* dev_;
};

}} // namespace ppl::llm

#endif
