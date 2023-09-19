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

#include "models/config.h"
#include "models/resource.h"
#include "models/llama/llama_tokenizer.h"
#include "models/llama/llama_worker.h"
#include "serving/grpc_server.h"
#include "backends/cuda/sampler.h"
#include "backends/cuda/resource_manager.h"
#include "common/processor.h"
#include "utils/index_manager.h"
#include "utils/queue_request_scheduler.h"
#include "utils/utils.h"
#include "utils/tokenizer.h"
#include "utils/config_utils.h"

using namespace std;
using namespace ppl::llm;
using namespace ppl::common;
using namespace ppl::nn;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "usage: " << argv[0] << " server_config.json" << endl;
        return -1;
    }

    ServerConfig server_config;
    if (!ppl::llm::utils::ParseServerConfig(argv[1], &server_config)) {
        LOG(ERROR) << "ParseServerConfig failed, server config file: " << argv[1];
        return -1;
    }

    WorkerConfig worker_config;
    worker_config.top_p = server_config.top_p;
    worker_config.top_k = server_config.top_k;
    worker_config.max_running_batch = server_config.max_running_batch;
    worker_config.max_tokens_per_request = server_config.max_tokens_per_request;
    worker_config.max_tokens_per_step = server_config.max_tokens_per_step;

    ModelConfig model_config;
    if (!ppl::llm::utils::ParseModelConfig(server_config.model_param_path, &model_config)) {
        LOG(ERROR) << "PaseModelConfig failed, model_param_path: " << server_config.model_param_path;
        return -1;
    }
    LOG(INFO) << "Parse model model_config successed";

    // init nccl, cuda engine, kv cache, kv scale manager
    cuda::CudaResourceManager resource_manager;
    auto rc = resource_manager.Init(model_config, server_config);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "init CudaResourceManager failed: " << GetRetCodeStr(rc);
        return -1;
    }
    std::shared_ptr<ppl::llm::utils::Tokenizer> tokenizer;

    if (server_config.model_type == "llama") {
        auto* llama_tokenizer = new llama::LlamaTokenizer();
        llama_tokenizer->Init(server_config.tokenizer_path);
        tokenizer = std::shared_ptr<ppl::llm::utils::Tokenizer>(llama_tokenizer);
    } else {
        LOG(ERROR) << "not supported model: " << server_config.model_type;
        return -1;
    }

    Resource resource;
    resource.tensor_parallel_size = server_config.tensor_parallel_size;
    resource.kv_cache_max_tokens = resource_manager.kv_cache_max_tokens;
    resource.items = resource_manager.items.data();
    resource.sampler = resource_manager.sampler.get();
    resource.device_worker_pool = &resource_manager.device_worker_pool;
    resource.tokenizer = tokenizer.get();

    GRPCServer svr;
    auto listen_addr = server_config.host + ":" + std::to_string(server_config.port);
    rc = svr.Init(listen_addr);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "GRPCConnection init failed.";
        return -1;
    }

    std::shared_ptr<RequestProcessor> llm_worker;
    if (server_config.model_type == "llama") {
        auto* llama_worker = new llama::LLaMAWorker(resource, model_config, worker_config);
        rc = llama_worker->Init();
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "llama_worker init failed: " << GetRetCodeStr(rc);
            return -1;
        }
        LOG(INFO) << "Init llama worker successed";
        svr.SetOnDisconnectedFunc([&llama_worker](Connection* c) {
            llama_worker->ClearTask(c);
        });
        llm_worker = std::shared_ptr<RequestProcessor>(llama_worker);
    } else {
        LOG(ERROR) << "not supported model: " << server_config.model_type;
        return -1;
    }

    LOG(INFO) << "listening on [" << listen_addr << "]";

    svr.Loop(llm_worker.get());

    return 0;
}
