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

#ifndef __PPL_LLM_CUDA_RESOURCE_MANAGER_H__
#define __PPL_LLM_CUDA_RESOURCE_MANAGER_H__

#include "models/config.h"
#include "models/resource.h"
#include "utils/sampler.h"

#include "ppl/common/log.h"
#include "ppl/common/barrier.h"
#include "ppl/common/threadpool.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/engines/llm_cuda/engine_factory.h"
#include "ppl/nn/runtime/runtime.h"

#ifdef PPLNN_CUDA_ENABLE_NCCL
#include "nccl.h"
#endif

#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace ppl { namespace llm { namespace cuda {

struct CudaResourceManager final {
    ~CudaResourceManager() {
        sampler.reset();
        
        for (auto it = items.begin(); it != items.end(); ++it) {
            cudaFree(it->kv_cache_mem);
            cudaFree(it->kv_scale_mem);
            delete it->runtime;
        }

        engine_list.clear();

#ifdef PPLNN_CUDA_ENABLE_NCCL
        for (auto it = nccl_comm_list.begin(); it != nccl_comm_list.end(); ++it) {
            auto e = ncclCommDestroy(*it);
            if (e != ncclSuccess) {
                LOG(ERROR) << "NCCL error(code:" << (int)e << ") on " << "(ncclCommDestroy)";
            }
        }
#endif

    }

    std::unique_ptr<ppl::llm::utils::Sampler> CreateCudaSampler(ppl::nn::Runtime* runtime);
    ppl::common::RetCode Init(const ModelConfig& model_config, const ServerConfig& server_config);
    ppl::common::StaticThreadPool device_worker_pool;
    std::vector<std::unique_ptr<ppl::nn::Engine>> engine_list;
    std::vector<ppl::llm::ResourceItem> items;
    std::unique_ptr<ppl::llm::utils::Sampler> sampler;
    uint64_t kv_cache_max_tokens;

#ifdef PPLNN_CUDA_ENABLE_NCCL
    std::vector<ncclComm_t> nccl_comm_list;
#endif

};

}}} // namespace ppl::llm::cuda

#endif