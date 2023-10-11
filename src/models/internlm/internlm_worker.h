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

#ifndef __PPL_LLM_INTERNLM_WORKER_H__
#define __PPL_LLM_INTERNLM_WORKER_H__

#include "common/processor.h"
#include "models/config.h"
#include "models/resource.h"
#include "models/llama/llama_worker.h"
#include "utils/index_manager.h"
#include "utils/queue_request_scheduler.h"
#include "utils/sampler.h"
#include "utils/tokenizer.h"

#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#include "ppl/nn/runtime/tensor.h"
#include "ppl/common/threadpool.h"

#include <memory>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace ppl { namespace llm { namespace internlm {

class InternLMWorker final : public RequestProcessor {
public:
    InternLMWorker(const Resource& resource, const ModelConfig& mconfig, const WorkerConfig& wconfig)
        : worker_(resource, mconfig, wconfig) {}

    ~InternLMWorker() {
        worker_.~LLaMAWorker();
    }

    ppl::common::RetCode Init();

    void ClearTask(Connection* conn) override {
        worker_.ClearTask(conn);
    }

    void Process(const std::shared_ptr<Request>& req, Connection* conn) {
        worker_.Process(req, conn);
    }

private:
    llama::LLaMAWorker worker_;
};

}}} // namespace ppl::llm::internlm

#endif