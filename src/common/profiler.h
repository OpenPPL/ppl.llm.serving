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
