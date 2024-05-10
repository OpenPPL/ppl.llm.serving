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

namespace ppl { namespace llm {

struct Profiler final {
    int step = 0;

    int finished_cnt = 0;
    int gen_token_cnt = 0;

    int kv_rest_blk = 0;
    int kv_max_blk = 0;

    int running_batch = 0;
    int max_running_batch = 0;
    int pending_task_size = 0;

    double dev_mem_total = 0.0;
    double dev_mem_free = 0.0;

    double prepare_duration = 0.0;
    double model_duration = 0.0;
    double sampling_duration = 0.0;
    double total_duration = 0.0;
    double set_input_duration = 0.0;
    double send_duration = 0.0;
    double early_finish_duration = 0.0;
    double penalty_duration = 0.0;

    double step_prepare_duration = 0.0;
    double step_set_input_duration = 0.0;
    double step_model_duration = 0.0;
    double step_penalty_duration = 0.0;
    double step_sampling_duration = 0.0;
    double step_send_duration = 0.0;
    double step_early_finish_duration = 0.0;
    double step_total_duration = 0.0;
};

}}

#endif
