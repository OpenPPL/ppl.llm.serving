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

#ifndef __PPL_LLM_PROFILER_H__
#define __PPL_LLM_PROFILER_H__

#include <memory>
#include <chrono>
#include <stdint.h>

namespace ppl { namespace llm {

struct ServerCounter final {
    // micro seconds
    uint64_t request_arrived_cnt = 0; // sum

    uint64_t deliver_task_step_cnt = 0;
    uint64_t deliver_task_step_cost = 0;

    uint64_t first_token_cnt = 0;
    uint64_t first_token_cost = 0;

    uint64_t generate_token_cnt = 0; // tps=（generated_token_cnt + first_token_cnt) / total_time
    uint64_t generate_token_cost = 0; // only including decoding procedure

    uint64_t total_cnt = 0; // finished_task count, used to compute qps=total_cnt / total_time
    uint64_t total_cost = 0;

    uint64_t parse_input_cnt = 0;
    uint64_t parse_input_cost = 0;
};

struct GeneratorReqCounter final {
    // micro seconds
    uint64_t encode_cnt = 0;
    uint64_t encode_cost = 0;

    uint64_t output_tokens_per_req = 0;

    char padding[40];   // avoid false sharing

    uint64_t waiting_cnt = 0;
    uint64_t waiting_cost = 0;
};

struct WorkerPerStepCounter {
    struct {
        uint64_t step_cnt = 0;
        uint64_t prepare_cost = 0;
        uint64_t set_input_cost = 0;
        uint64_t model_forward_cost = 0;
        uint64_t choose_token_cost = 0; // penalty + sampling
        uint64_t post_process_cost = 0;
        uint64_t total_cost = 0;
        uint64_t input_token_cnt = 0; // per step
        uint64_t output_token_cnt = 0;
        uint64_t cache_hit_count = 0;
    } global, current;
};

struct WorkerProfiler {
    uint64_t finished_task_cnt = 0;

    uint64_t kv_rest_blk = 0;
    uint64_t kv_max_blk = 0;

    uint64_t running_task = 0;
    uint64_t prefill_batch = 0;
    uint64_t prefill_tokens = 0;

    uint64_t max_running_task = 0;
    uint64_t pending_task_size = 0;

    uint64_t dev_mem_total = 0;
    uint64_t dev_mem_free = 0;

    WorkerPerStepCounter step_counter;
    GeneratorReqCounter req_counter;
};

struct ServerProfiler final {

    ServerProfiler& operator=(const ServerProfiler& other) {
        if (this == &other) {
            return *this;
        }
        // deep copy
        *worker_profiler = *other.worker_profiler;
        server_counter = other.server_counter;
        qps = other.qps;
        tps = other.tps;
        global_time_cost = other.global_time_cost;
        return *this;
    }

    std::shared_ptr<WorkerProfiler> worker_profiler;
    ServerCounter server_counter;
    double qps = 0; // qps=finished_cnt / global_time_cost
    double tps = 0; // tps=generate_token_cnt / global_time_cost
    uint64_t global_time_cost;
};

void PrintProfiler(const WorkerProfiler& worker_profiler);

}} // namespace ppl::llm

#endif
