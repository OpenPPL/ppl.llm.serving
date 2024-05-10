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

#include "profiling_utils.h"
#include <cstdio>

namespace ppl { namespace llm { namespace utils {

void PrintProfiler(const Profiler& profiler) {
    fprintf(stderr, "[PERF] --- step %d -------------------------------------------------\n", profiler.step);
    fprintf(stderr, "[PERF]  |- memory usage: (%.2f - %.2f) -> %.2f GiB\n", profiler.dev_mem_total, profiler.dev_mem_free,
            profiler.dev_mem_total - profiler.dev_mem_free);
    fprintf(stderr, "[PERF]  |- kv cache usage: %.2f %%\n", (1.0f - (double)profiler.kv_rest_blk / profiler.kv_max_blk) * 100.0);
    fprintf(stderr, "[PERF]  |- pending task number: %d\n", profiler.pending_task_size);
    fprintf(stderr, "[PERF]  |- running batch: %d, max running batch: %d\n", profiler.running_batch, profiler.max_running_batch);
    fprintf(stderr, "[PERF]  |- finished query count: %d, QPS: %.2f\n", profiler.finished_cnt,
            float(profiler.finished_cnt) / profiler.total_duration * 1000);
    fprintf(stderr, "[PERF]  |- gen token count: %d, avg gen len: %.2f, TPS: %.2f\n", profiler.gen_token_cnt,
            profiler.finished_cnt ? profiler.gen_token_cnt / float(profiler.finished_cnt) : 0.0f,
            float(profiler.gen_token_cnt) / profiler.total_duration * 1000);

    fprintf(stderr, "[PERF]  |- pipeline          | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_total_duration, profiler.total_duration / profiler.step, profiler.total_duration);
    fprintf(stderr, "[PERF]  |-- batching         | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_prepare_duration, profiler.prepare_duration / profiler.step, profiler.prepare_duration);
    fprintf(stderr, "[PERF]  |-- copy inputs      | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_set_input_duration, profiler.set_input_duration / profiler.step, profiler.set_input_duration);
    fprintf(stderr, "[PERF]  |-- model inference  | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_model_duration, profiler.model_duration / profiler.step, profiler.model_duration);
    fprintf(stderr, "[PERF]  |-- penalty         | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_penalty_duration, profiler.penalty_duration / profiler.step, profiler.penalty_duration);
    fprintf(stderr, "[PERF]  |-- sampling         | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_sampling_duration, profiler.sampling_duration / profiler.step, profiler.sampling_duration);
    fprintf(stderr, "[PERF]  |-- send response    | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_send_duration, profiler.send_duration / profiler.step, profiler.send_duration);
    fprintf(stderr, "[PERF]  |-- early finish     | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            profiler.step_early_finish_duration, profiler.early_finish_duration / profiler.step, profiler.early_finish_duration);

    fprintf(stderr, "[PERF]  |- schedule cost: %.2f %%\n",
            (profiler.total_duration - profiler.model_duration) / profiler.total_duration * 100);
}

}}}
