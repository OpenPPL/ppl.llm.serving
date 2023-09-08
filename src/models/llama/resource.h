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

#ifndef __PPL_LLM_RESOURCE_H__
#define __PPL_LLM_RESOURCE_H__

#include "ppl/common/threadpool.h"
#include "ppl/nn/runtime/runtime.h"
#include <vector>
#include <memory>

namespace ppl { namespace llm { namespace llama {

struct ResourceItem final {
    void* kv_cache_mem = nullptr;
    void* kv_scale_mem = nullptr;
    ppl::nn::Runtime* runtime = nullptr;
};

struct Resource final {
    uint32_t tensor_parallel_size = 0;
    uint64_t kv_cache_max_tokens = 0;
    ResourceItem* items = nullptr;
    ppl::common::ThreadPool* device_worker_pool = nullptr;
};

}}} // namespace ppl::llm::llama

#endif
