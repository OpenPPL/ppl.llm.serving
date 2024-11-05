#include "profiler.h"
#include "stdio.h"

namespace ppl { namespace llm {

void PrintProfiler(const WorkerProfiler& worker_profiler) {
    float qps = float(worker_profiler.finished_task_cnt) / worker_profiler.step_counter.global.total_cost * 1e6;
    float tps = float(worker_profiler.step_counter.global.output_token_cnt) /
        worker_profiler.step_counter.global.total_cost * 1e6;
    float cache_hit_rate = float(worker_profiler.step_counter.global.cache_hit_count) /
        worker_profiler.step_counter.global.input_token_cnt;

    fprintf(stderr, "[PERF] --- step %ld -------------------------------------------------\n",
            worker_profiler.step_counter.global.step_cnt);
    fprintf(stderr, "[PERF]  |- memory usage: (%.2f - %.2f) -> %.2f GiB\n", float(worker_profiler.dev_mem_total) / 1e9,
            float(worker_profiler.dev_mem_free) / 1e9,
            float(worker_profiler.dev_mem_total - worker_profiler.dev_mem_free) / 1e9);
    fprintf(stderr, "[PERF]  |- kv cache usage: %.2f %%\n",
            (1.0f - (float)worker_profiler.kv_rest_blk / worker_profiler.kv_max_blk) * 100.0);
    fprintf(stderr, "[PERF]  |- pending task number: %ld\n", worker_profiler.pending_task_size);
    fprintf(stderr, "[PERF]  |- running batch: %ld, max running batch: %ld\n", worker_profiler.running_task,
            worker_profiler.max_running_task);
    fprintf(stderr, "[PERF]  |- prefill batch: %ld , prefill tokens: %ld\n", worker_profiler.prefill_batch,
            worker_profiler.prefill_tokens);
    fprintf(stderr, "[PERF]  |- prefix cache hit rate: %.2f %%\n", cache_hit_rate * 100);
    fprintf(stderr, "[PERF]  |- finished query count: %ld, QPS: %.2f\n", worker_profiler.finished_task_cnt, qps);
    fprintf(stderr, "[PERF]  |- gen token count: %ld, avg gen len: %.2f, TPS: %.2f\n",
            worker_profiler.step_counter.global.output_token_cnt,
            worker_profiler.finished_task_cnt
                ? worker_profiler.step_counter.global.output_token_cnt / float(worker_profiler.finished_task_cnt)
                : 0.0f,
            tps);

    fprintf(stderr, "[PERF]  |- pipeline          | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            float(worker_profiler.step_counter.current.total_cost) / 1e3,
            float(worker_profiler.step_counter.global.total_cost / 1e3) / worker_profiler.step_counter.global.step_cnt,
            float(worker_profiler.step_counter.global.total_cost) / 1e3);
    fprintf(
        stderr, "[PERF]  |-- batching         | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
        float(worker_profiler.step_counter.current.prepare_cost) / 1e3,
        float(worker_profiler.step_counter.global.prepare_cost) / 1e3 / worker_profiler.step_counter.global.step_cnt,
        float(worker_profiler.step_counter.global.prepare_cost) / 1e3);
    fprintf(
        stderr, "[PERF]  |-- set inputs       | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
        float(worker_profiler.step_counter.current.set_input_cost) / 1e3,
        float(worker_profiler.step_counter.global.set_input_cost) / 1e3 / worker_profiler.step_counter.global.step_cnt,
        float(worker_profiler.step_counter.global.set_input_cost) / 1e3);
    fprintf(stderr, "[PERF]  |-- model inference  | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            float(worker_profiler.step_counter.current.model_forward_cost) / 1e3,
            float(worker_profiler.step_counter.global.model_forward_cost) / 1e3 /
                worker_profiler.step_counter.global.step_cnt,
            float(worker_profiler.step_counter.global.model_forward_cost) / 1e3);
    fprintf(stderr, "[PERF]  |-- choose token     | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            float(worker_profiler.step_counter.current.choose_token_cost) / 1e3,
            float(worker_profiler.step_counter.global.choose_token_cost) / 1e3 /
                worker_profiler.step_counter.global.step_cnt,
            float(worker_profiler.step_counter.global.choose_token_cost) / 1e3);
    fprintf(stderr, "[PERF]  |-- post process     | cur: %.2f ms, | avg: %.2f ms, | total: %.2f ms\n",
            float(worker_profiler.step_counter.current.post_process_cost) / 1e3,
            float(worker_profiler.step_counter.global.post_process_cost) / 1e3 /
                worker_profiler.step_counter.global.step_cnt,
            float(worker_profiler.step_counter.global.post_process_cost) / 1e3);
    fprintf(
        stderr, "[PERF]  |- schedule cost: %.2f %%\n",
        float(worker_profiler.step_counter.global.total_cost - worker_profiler.step_counter.global.model_forward_cost -
              worker_profiler.step_counter.global.choose_token_cost) /
            worker_profiler.step_counter.global.total_cost * 100);
}

}} // namespace ppl::llm