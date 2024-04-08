#include "simple_flags.h"

#include "llm.grpc.pb.h"
#include "ppl/common/log.h"
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "grpc++/grpc++.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <random>

using namespace grpc;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using namespace std::chrono;
using namespace ppl::llm;

Define_string_opt("--target", g_flag_target, "localhost:23333", "ip:port");
Define_string_opt("--dataset", g_flag_dataset, "", "Path to the dataset.");
Define_string_opt("--request_rate", g_flag_request_rate, "inf",
                  "Number of request per second. If this is inf, then all the requests are sent at time 0. Otherwise, "
                  "we use Poisson process to synthesize the request arrival times.");
Define_bool_opt("--early_stopping", g_flag_early_stopping, true,
                  "whether enable early stopping when met end token id");

struct TidRecord {
    int prompt_len;
    int exp_output_len;
    int real_output_len = 0;
    bool is_prefill = true;
    std::chrono::_V2::system_clock::time_point send_time;
    std::chrono::_V2::system_clock::time_point prefill_time;
    std::chrono::_V2::system_clock::time_point finished_time;
    std::chrono::_V2::system_clock::time_point prev_time;

};

static std::unordered_map<int, std::vector<int>> g_rsp_stream_store;
static std::vector<double> g_decode_latecy_list; 
static int g_finished_cnt = 0;
static int g_num_request = 0;
static std::unordered_map<int64_t, TidRecord> g_tid_record_map;

void SampleRequest(const std::string& dataset_path, bool early_stopping, std::vector<std::shared_ptr<proto::BatchedRequest>>* req_list) {
    std::ifstream ifs(dataset_path);
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document root;
    root.ParseStream(isw);
    if (root.HasParseError()) {
        return;
    }
    LOG(INFO) << "request size: " << root.Size();
    uint64_t tid = 0;
    for (size_t i = 0; i < root.Size(); ++i) {
        const auto& input_token_ids = root[i]["input_token_ids"].GetArray();
        const int gen_len = 200;
        auto batch_req = std::make_shared<proto::BatchedRequest>(); // batch_size = 1
        auto* req = batch_req->add_req();
        req->set_id(tid);
        req->set_temperature(1);
        auto* pb_tokens = req->mutable_tokens();
        for (auto it = input_token_ids.begin(); it != input_token_ids.end(); ++it) {
            pb_tokens->add_ids(it->GetInt());
        }

        auto* stopping_parameters = req->mutable_stopping_parameters();
        stopping_parameters->set_max_new_tokens(gen_len);
        stopping_parameters->set_ignore_eos_token(!early_stopping);
        req_list->push_back(batch_req);

        auto& tid_record = g_tid_record_map.emplace(tid, TidRecord()).first->second;
        tid_record.prompt_len = input_token_ids.Size();
        tid_record.exp_output_len = gen_len;
        ++tid;
    }
}

enum CallStatus { CREATE, PROCESS, PROCESSED, FINISH, FAILED };

class GenerationClientAsync {
public:
    GenerationClientAsync(std::shared_ptr<Channel> channel) : stub_(proto::LLMService::NewStub(channel)) {}

    void Generation(const std::vector<std::shared_ptr<proto::BatchedRequest>> req_list) {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t i = 0; i < req_list.size(); i++) {
            const auto& req_batch = *req_list[i];

            auto it = g_tid_record_map.find(req_batch.req(0).id());
            if (it == g_tid_record_map.end()) {
                LOG(ERROR) << "unrecoginized tid: " << req_batch.req(0).id();
                return;
            }
            it->second.send_time = std::chrono::high_resolution_clock::now();

            AsyncClientCall* call = new AsyncClientCall;

            call->response_reader = stub_->PrepareAsyncGeneration(&call->context, req_batch, &cq_);
            call->response_reader->StartCall((void*)call);

            if (g_flag_request_rate == "inf") { // continuous send, no interval
                continue;
            }
            float request_rate = std::stof(g_flag_request_rate);
            std::exponential_distribution<> dist(request_rate);
            float sleep_time = dist(gen);
            int sleep_s = int(sleep_time);
            int sleep_us = (sleep_time - float(sleep_s)) * 1000000;
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_s * 1000000 + sleep_us));
        }
        pthread_mutex_lock(&lock_);
        while (g_finished_cnt < g_num_request) {
            pthread_cond_wait(&finished_cond_, &lock_);
        }
        pthread_mutex_unlock(&lock_);
    }

    // Loop while listening for completed responses.
    void AsyncCompleteRpc() {
        void* got_tag;
        bool ok = false;
        // Block until the next result is available in the completion queue "cq".
        LOG(INFO) << "Wait for response";
        while (cq_.Next(&got_tag, &ok)) {
            if (!got_tag) {
                LOG(ERROR) << "Get tag failed";
            }

            // The tag in this example is the memory location of the call object
            AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);
            call->HandleResponse(ok);

            if (g_finished_cnt >= g_num_request) {
                pthread_cond_signal(&finished_cond_);
                break;
            }
        }
    }

private:
    struct AsyncClientCall {
        void HandleResponse(bool responseStatus) {
            switch (callStatus_) {
                case CREATE:
                    if (responseStatus) {
                        response_reader->Read(&reply, (void*)this);
                        callStatus_ = PROCESS;
                    } else {
                        response_reader->Finish(&status, (void*)this);
                        callStatus_ = FINISH;
                    }
                    break;
                case PROCESS:
                    if (responseStatus) {
                        const auto& batched_rsp = this->reply;
                        for (const auto& rsp : batched_rsp.rsp()) {
                            int tid = rsp.id();
                            if (g_tid_record_map[tid].is_prefill == true) {
                                g_tid_record_map[tid].prefill_time = std::chrono::high_resolution_clock::now();
                                g_tid_record_map[tid].is_prefill = false;
                                g_tid_record_map[tid].prev_time = g_tid_record_map[tid].prefill_time;
                            } else {
                                auto cur_time = std::chrono::high_resolution_clock::now();
                                double step_latency = std::chrono::duration_cast<std::chrono::microseconds>(cur_time - g_tid_record_map[tid].prev_time).count() / 1000.0; // ms
                                g_decode_latecy_list.push_back(step_latency);
                            }
                            int rsp_token = rsp.tokens().ids().at(0);
                            g_rsp_stream_store[tid].push_back(rsp_token);
                            g_tid_record_map[tid].real_output_len += 1;
                            response_reader->Read(&reply, (void*)this);
                        }
                    } else {
                        response_reader->Finish(&status, (void*)this);
                        callStatus_ = FINISH;
                    }
                    break;
                case FINISH: 
                    __sync_fetch_and_add(&g_finished_cnt, 1);
                    if (status.ok()) {
                        for (const auto& rsp : this->reply.rsp()) {
                            g_tid_record_map[rsp.id()].finished_time = std::chrono::high_resolution_clock::now();
                            LOG(INFO) << "Finish: " << g_finished_cnt << "/" << g_num_request;
                            LOG(INFO) << "Server Response Completed: " << rsp.id();
                        }                        
                    } else {
                        LOG(ERROR) << "RPC failed";
                    }
                    delete this;
                    break;
                default:
                    LOG(ERROR) << "impossible or invalid status";
                    break;
            }
        };

        CallStatus callStatus_ = CREATE;
        proto::BatchedResponse reply;
        ClientContext context;
        Status status;
        std::unique_ptr<ClientAsyncReader<proto::BatchedResponse>> response_reader;
    };

    std::unique_ptr<proto::LLMService::Stub> stub_;

    CompletionQueue cq_;
    pthread_cond_t finished_cond_ = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;
};

int main(int argc, char* argv[]) {
    simple_flags::parse_args(argc, argv);
    if (!simple_flags::get_unknown_flags().empty()) {
        string content;
        for (auto it : simple_flags::get_unknown_flags()) {
            content += "'" + it + "', ";
        }
        content.resize(content.size() - 2); // remove last ', '
        content.append(".");
        LOG(ERROR) << "unknown option(s): " << content.c_str();
        return -1;
    }

    const std::string target_str = g_flag_target;
    const std::string data_path = g_flag_dataset; // samples_1024.json

    std::vector<std::shared_ptr<proto::BatchedRequest>> req_list;
    SampleRequest(data_path, g_flag_early_stopping, &req_list);
    g_num_request = req_list.size();

    GenerationClientAsync generator(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));

    std::thread recv_thread = std::thread(&GenerationClientAsync::AsyncCompleteRpc, &generator);

    auto benchmark_start = std::chrono::high_resolution_clock::now();
    generator.Generation(req_list);
    auto benchmark_end = std::chrono::high_resolution_clock::now();

    auto benchmark_time =
        double(std::chrono::duration_cast<std::chrono::microseconds>(benchmark_end - benchmark_start).count()) /
        1000.0 / 1000.0;

    double total_prefill_latency = 0; // ms
    double total_decode_latency_per_token = 0; // ms
    double total_prompt_latency = 0; // ms
    int total_input_tokens = 0;
    int total_gen_tokens = 0;
    int total_exp_gen_tokens = 0;
    int max_input_len = 0, min_input_len = INT32_MAX;
    std::vector<double> prefill_latency_list, prompt_latency_list;
    for (auto it = g_tid_record_map.begin(); it != g_tid_record_map.end(); ++it) {
        auto& tid_record = it->second;
        max_input_len = std::max(tid_record.prompt_len, max_input_len);
        min_input_len = std::min(tid_record.prompt_len, min_input_len);
        double prefill_latency = double(
            std::chrono::duration_cast<std::chrono::microseconds>(tid_record.prefill_time - tid_record.send_time).count() /
            1000.0); // ms

        double decoding_latency = double(
            std::chrono::duration_cast<std::chrono::microseconds>(tid_record.finished_time - tid_record.prefill_time)
                .count() /
            1000.0); // ms
        double prompt_latency = double(
            std::chrono::duration_cast<std::chrono::microseconds>(tid_record.finished_time - tid_record.send_time).count() /
            1000.0); // ms
        // total_latency_per_token += (prompt_latency / tid_record.real_output_len);
        total_prompt_latency += prompt_latency;

        total_prefill_latency += prefill_latency;

        total_decode_latency_per_token += tid_record.real_output_len > 1 ? decoding_latency / (tid_record.real_output_len - 1) : 0.0f;

        total_input_tokens += tid_record.prompt_len;
        total_exp_gen_tokens += tid_record.exp_output_len;
        total_gen_tokens += tid_record.real_output_len;

        prefill_latency_list.push_back(prefill_latency);
        prompt_latency_list.push_back(prompt_latency);
    }
    double avg_latency_prefill = total_prefill_latency / g_num_request;
    double avg_latency_decode_per_token = total_decode_latency_per_token / g_num_request;
    double avg_latency_per_prompt = total_prompt_latency / g_num_request;

    std::sort(prefill_latency_list.begin(), prefill_latency_list.end());
    std::sort(g_decode_latecy_list.begin(), g_decode_latecy_list.end());
    std::sort(prompt_latency_list.begin(), prompt_latency_list.end());

    fprintf(stderr, "[INPUT] dataset: %s, request_rate %s, early_stopping: %d\n", g_flag_dataset.c_str(), g_flag_request_rate.c_str(), g_flag_early_stopping);
    fprintf(stderr, "[RESULT] benchmark time: %.2f s\n", benchmark_time);

    // 统计: avg inptu len, avg gen len, task num, total gen tokens
    fprintf(stderr, "[RESULT] request count: %d\n", g_num_request);
    fprintf(stderr, "[RESULT] avg input len: %d, max input len: %d, min input len: %d, total input len: %d\n", total_input_tokens / g_num_request, max_input_len, min_input_len,
            total_input_tokens);
    fprintf(stderr, "[RESULT] avg gen len: %d, real total gen len: %d, expected total gen len: %d\n", total_gen_tokens / g_num_request, total_gen_tokens, total_exp_gen_tokens);
    fprintf(stderr, "[RESULT] time per token: %.2f ms\n", benchmark_time * 1000 / total_gen_tokens);
    fprintf(stderr, "[RESULT] avg latency prefill: %.2f ms\n", avg_latency_prefill);
    fprintf(stderr, "[RESULT] avg latency decoding: %.2f ms\n", avg_latency_decode_per_token);
    fprintf(stderr, "[RESULT] avg latency per prompt: %.2f ms\n", avg_latency_per_prompt);

    // tps1, tps2
    fprintf(stderr, "[RESULT] tokens out per sec: %.2f\n", total_gen_tokens / benchmark_time);
    fprintf(stderr, "[RESULT] tokens inout per sec: %.2f\n", (total_input_tokens + total_gen_tokens) / benchmark_time);
    // qps
    fprintf(stderr, "[RESULT] requests per sec: %.2f\n", g_num_request / benchmark_time);

    // distribution
    fprintf(stderr, "[RESULT] prefill latency distribution (ms): \n    min:[%.2f], 1%%[%.2f], 10%%:[%.2f], 25%%:[%.2f], 50%%:[%.2f], 75%%:[%.2f], 80%%:[%.2f], 90%%:[%.2lf], 95%%[%.2f], 99%%[%.2f], max:[%.2f]\n",
        prefill_latency_list[0], prefill_latency_list[g_num_request / 100], prefill_latency_list[g_num_request / 10], prefill_latency_list[g_num_request / 4],
        prefill_latency_list[g_num_request / 2], prefill_latency_list[g_num_request * 3 / 4], prefill_latency_list[g_num_request * 8 / 10],
        prefill_latency_list[g_num_request * 9 / 10], prefill_latency_list[g_num_request * 95 / 100], prefill_latency_list[g_num_request * 99 / 100], prefill_latency_list[g_num_request - 1]);

    fprintf(stderr, "[RESULT] decode latency per token distribution (ms): \n    min:[%.2f], 1%%[%.2f], 10%%:[%.2f], 25%%:[%.2f], 50%%:[%.2f], 75%%:[%.2f], 80%%:[%.2f], 90%%:[%.2lf], 95%%[%.2f], 99%%[%.2f], max:[%.2f]\n",
        g_decode_latecy_list[0], g_decode_latecy_list[g_num_request / 100], g_decode_latecy_list[g_num_request / 10], g_decode_latecy_list[g_num_request / 4],
        g_decode_latecy_list[g_num_request / 2], g_decode_latecy_list[g_num_request * 3 / 4], g_decode_latecy_list[g_num_request * 8 / 10],
        g_decode_latecy_list[g_num_request * 9 / 10], g_decode_latecy_list[g_num_request * 95 / 100], g_decode_latecy_list[g_num_request * 99 / 100], g_decode_latecy_list[g_num_request - 1]);

    fprintf(stderr, "[RESULT] prompt latency distribution (ms): \n    min:[%.2f], 1%%[%.2f], 10%%:[%.2f], 25%%:[%.2f], 50%%:[%.2f], 75%%:[%.2f], 80%%:[%.2f], 90%%:[%.2lf], 95%%[%.2f], 99%%[%.2f], max:[%.2f]\n",
        prompt_latency_list[0], prompt_latency_list[g_num_request / 100], prompt_latency_list[g_num_request / 10], prompt_latency_list[g_num_request / 4],
        prompt_latency_list[g_num_request / 2], prompt_latency_list[g_num_request * 3 / 4], prompt_latency_list[g_num_request * 8 / 10],
        prompt_latency_list[g_num_request * 9 / 10], prompt_latency_list[g_num_request * 95 / 100], prompt_latency_list[g_num_request * 99 / 100], prompt_latency_list[g_num_request - 1]);

    fprintf(stdout, "CSV format header:total_latency,qps,avg_latency,min,1%%,10%%,25%%,50%%,75%%,80%%,90%%,95%%,99%%,max");
    fprintf(stdout, "CSV format output: %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f", benchmark_time, g_num_request / benchmark_time, avg_latency_per_prompt, 
        g_decode_latecy_list[0], g_decode_latecy_list[g_num_request / 100], g_decode_latecy_list[g_num_request / 10], g_decode_latecy_list[g_num_request / 4],
        g_decode_latecy_list[g_num_request / 2], g_decode_latecy_list[g_num_request * 3 / 4], g_decode_latecy_list[g_num_request * 8 / 10],
        g_decode_latecy_list[g_num_request * 9 / 10], g_decode_latecy_list[g_num_request * 95 / 100], g_decode_latecy_list[g_num_request * 99 / 100], g_decode_latecy_list[g_num_request - 1]);

    recv_thread.join();
    return 0;
}
