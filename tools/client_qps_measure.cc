#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <chrono>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "llm.grpc.pb.h"
#include <grpc++/grpc++.h>
#include <chrono>
#include <sentencepiece_processor.h>
#include <fstream>
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "ppl/common/log.h"

using namespace grpc;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using namespace std::chrono;
using namespace ppl::llm;

static pthread_cond_t has_request = PTHREAD_COND_INITIALIZER;    //初始化条件变量
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;        //初始化互斥锁
std::vector<std::shared_ptr<proto::Request>> req_list;
static std::unordered_map<int64_t, std::shared_ptr<proto::Request>> tid_req_map;
static std::unordered_map<int, std::string> rsp_stream_store;


struct TidRecord {
    int prompt_len;
    int output_len;
    std::chrono::_V2::system_clock::time_point finished_time;
};

static std::unordered_map<int64_t, TidRecord> tid_record_map;  // prompt_len, output_len, latency


static int finished_cnt = 0;
static int num_request = 0;

static int total_input_tokens;
static int total_gen_tokens;

void SampleRequest(
    const std::string& dataset_path, 
    const sentencepiece::SentencePieceProcessor& tokenizer) {

    std::ifstream ifs(dataset_path);
    rapidjson::IStreamWrapper isw(ifs);
    rapidjson::Document root;
    root.ParseStream(isw);
    if (root.HasParseError()) {
        return;
    }
    LOG(INFO) << "root.size()" << root.Size();
    uint64_t tid = 0;
    for (size_t i=0; i<root.Size(); ++i) {
        const auto& convs = root[i]["conversations"];

        // Filter out the conversations with less than 2 turns.
        if (convs.Size() < 2) {
            continue;
        }

        const std::string prompt = convs[0]["value"].GetString();
        const std::string ans = convs[1]["value"].GetString();


        std::vector<int> prompt_token_ids;
        std::vector<int> ans_token_ids;
        tokenizer.Encode(prompt, &prompt_token_ids);
        tokenizer.Encode(ans, &ans_token_ids);

        // filter out too long or too short seq
        bool too_short = prompt_token_ids.size() < 4 || ans_token_ids.size() < 4;
        bool too_long = prompt_token_ids.size() > 1024 || prompt_token_ids.size() + ans_token_ids.size() > 2048;

        total_input_tokens += prompt_token_ids.size();
        total_gen_tokens += ans_token_ids.size();

        auto req = std::make_shared<proto::Request>();
        req->set_id(tid);
        req->set_prompt(prompt);
        req->set_temperature(1);
        req->set_generation_length(ans_token_ids.size());

        req_list.push_back(req);
        tid_req_map.emplace(req->id(), req);

        auto& tid_record = tid_record_map.emplace(tid, TidRecord()).first->second;
        tid_record.prompt_len = prompt_token_ids.size();
        tid_record.output_len = ans_token_ids.size();
        // tid_record_map[tid].prompt_len = prompt_token_ids.size();
        // tid_record_map[tid].output_len = ans_token_ids.size();

        tid++;
    }
}

enum CallStatus {CREATE, PROCESS, PROCESSED, FINISH, FAILED};

class GenerationClientAsync {
public:
    GenerationClientAsync(std::shared_ptr<Channel> channel)
        : stub_(proto::LLMService::NewStub(channel)) {}

    void Generation() {
        while(finished_cnt < num_request) {
            pthread_mutex_lock(&lock);     
            while(req_list.size() == 0 && finished_cnt < num_request) {
                pthread_cond_wait(&has_request, &lock);
            }
            // LOG(INFO) << "req_list.size(): " << req_list.size();
            // std::cout << "req_list.size(): " << req_list.size() << std::endl;
            for(int i=0; i<req_list.size(); i++) {
                proto::BatchedRequest req_batch;
                auto* req = req_batch.add_req();
                req->set_id(req_list[i]->id());
                req->set_prompt(req_list[i]->prompt());
                req->set_temperature(req_list[i]->temperature());
                req->set_generation_length(req_list[i]->generation_length());

                AsyncClientCall* call = new AsyncClientCall;

                // LOG(INFO) << "tid: " << req_batch.req(0).id() << ", send prompt size: " << req_batch.req(0).prompt().size();
                // 创建RPC
                call->response_reader = stub_->PrepareAsyncGeneration(&call->context, req_batch, &cq_);
                // StartCall initiates the RPC call
                call->response_reader->StartCall((void*)call);

                // Request that, upon completion of the RPC, "reply" be updated with the
                // server's response; "status" with the indication of whether the operation
                // was successful. Tag the request with the memory address of the call
                // object.

                // call->response_reader->Finish(&call->status, (void*)call);    // really send requst
                // std::cout << "sent async call" << std::endl;
            }
            req_list.clear();
            pthread_mutex_unlock(&lock);   
            // std::this_thread::sleep_for(std::chrono::seconds(1));
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

  // Loop while listening for completed responses.
  // Prints out the response from the server.
    void AsyncCompleteRpc() {
        void* got_tag;
        bool ok = false;
        // Block until the next result is available in the completion queue "cq".
        LOG(INFO) << "Wait for response";
        while (cq_.Next(&got_tag, &ok)) {
            if (!got_tag) {
                LOG(ERROR) << "Get tag failed";
            }
            // std::cout << "get cp.Next" << std::endl;
            // The tag in this example is the memory location of the call object
            AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);
            call->HandleResponse(ok);

            if(finished_cnt >= num_request) {
                pthread_cond_signal(&has_request);
                break;
            }
        }
    }


private:
  struct AsyncClientCall {

    void HandleResponse(bool responseStatus) {

        switch (callStatus_) {
            case CREATE:
                // LOG(INFO) << "CREATE";
                if (responseStatus) {
                    response_reader->Read(&reply, (void*)this);
                    callStatus_ = PROCESS;
                } else {
                    response_reader->Finish(&status, (void*)this);
                    callStatus_ = FINISH;
                }           
                break;
            case PROCESS:
                // LOG(INFO) << "PROCESS";
                if (responseStatus) {
                    auto& rsp = this->reply;
                    int tid = rsp.id();
                    // 失败重发
                    if (rsp.status() == proto::FAILED) {
                        auto req = tid_req_map[tid];
                        pthread_mutex_lock(&lock);
                        req_list.push_back(req);
                        LOG(WARNING) << "Resend request id: " << tid;
                        pthread_mutex_unlock(&lock);
                        pthread_cond_signal(&has_request);
                        response_reader->Finish(&status, (void*)this);
                        callStatus_ = FAILED;
                    } else {
                        const std::string& rsp_stream = rsp.generated();
                        rsp_stream_store[tid] += (rsp_stream + " ");
                        // LOG(INFO) << "rsp_stream: " << rsp_stream;
                        response_reader->Read(&reply, (void*)this);
                    }

                } else {
                    response_reader->Finish(&status, (void*)this);
                    callStatus_ = FINISH;
                }
                break;
            case FAILED:
                delete this;
                break;
            case FINISH:
                __sync_fetch_and_add(&finished_cnt, 1);
                // auto it = tid_record_map.find(reply.id());
                // if (it == tid_record_map.end()) {
                //     LOG(ERROR) << "Find non exist tid: " << reply.id();
                // }
                // auto& tid_record = it->second;                tid_record.finished_time = std::chrono::high_resolution_clock::now();
                // tid_record.finished_time = std::chrono::high_resolution_clock::now();

                tid_record_map.find(reply.id())->second.finished_time = std::chrono::high_resolution_clock::now();
                 LOG(INFO) << "Finish: " << finished_cnt << "/"<< num_request;
                if (status.ok()) {
                    LOG(INFO) << "Server Response Completed: " << reply.id();
                }
                else {
                    LOG(ERROR) << "RPC failed";
                }
                // LOG(INFO) << rsp_stream_store[reply.id()];
                delete this;
                break;
            default:
                LOG(ERROR) << "impossible or invalid status";
                break;
        }

    };
    CallStatus callStatus_ = CREATE;

    // Container for the data we expect from the server.
    proto::Response reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // Storage for the status of the RPC upon completion.
    Status status;

    std::unique_ptr<ClientAsyncReader<proto::Response>> response_reader;
  };

  // Out of the passed in Channel comes the stub, stored here, our view of the
  // server's exposed services.
  std::unique_ptr<proto::LLMService::Stub> stub_;

  // The producer-consumer queue we use to communicate asynchronously with the
  // gRPC runtime.
  CompletionQueue cq_;
};

int main(int argc, char const *argv[]) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " host:port tokenizer_path samples_json_path" << std::endl;
        return -1;
    }
    const std::string target_str = argv[1];
    const std::string tokenizer_path = argv[2]; // LLaMA/tokenizer.model
    const std::string data_path = argv[3];  // ./samples_1024.json

    sentencepiece::SentencePieceProcessor tokenizer;
    const auto tokenizer_status = tokenizer.Load(tokenizer_path);
    if (!tokenizer_status.ok()) {
        LOG(ERROR) << tokenizer_status.ToString();
        return -1;
    }
    LOG(INFO) << "VOCAB_SIZE: " << tokenizer.GetPieceSize()
            << "; BOS ID: " << tokenizer.bos_id()
            << "; EOS ID: "<< tokenizer.eos_id()
            << "; PAD ID: " << tokenizer.pad_id();

    SampleRequest(data_path, tokenizer); 
    num_request = req_list.size();

    GenerationClientAsync generator(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));

    std::thread thread_ = std::thread(&GenerationClientAsync::AsyncCompleteRpc, &generator);
    
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    generator.Generation();
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    
    auto benchmark_time = double(std::chrono::duration_cast<std::chrono::microseconds>(benchmark_end - benchmark_start).count()) / 1000.0 / 1000.0;


    double total_latency_per_token = 0;   // ms
    double total_latency = 0;             // ms
    for (auto it=tid_record_map.begin(); it!=tid_record_map.end(); ++it) {
        int tid = it->first;
        auto& tid_record = it->second;
        double latency = double(std::chrono::duration_cast<std::chrono::microseconds>(tid_record.finished_time - benchmark_start).count() / 1000.0);  // ms
        total_latency_per_token += (latency / tid_record.output_len);
        total_latency += latency;
    }
    double avg_latency_per_prompt = total_latency / num_request / 1000;
    double avg_latency_per_token = total_latency_per_token / num_request / 1000;

    fprintf(stderr, "[RESULT] benchmark time: %.2f s\n", benchmark_time);

    // 统计
    // avg inptu len, avg gen len, task num, total gen tokens
    fprintf(stderr, "[RESULT] request count: %d\n", num_request);
    fprintf(stderr, "[RESULT] avg input len: %d, total input len: %d\n", total_input_tokens / num_request, total_input_tokens);
    fprintf(stderr, "[RESULT] avg gen len: %d, total gen len: %d\n", total_gen_tokens / num_request, total_gen_tokens);
    fprintf(stderr, "[RESULT] time per token: %.2f ms\n", benchmark_time * 1000 / total_gen_tokens);
    // tps1, tps2
    fprintf(stderr, "[RESULT] tokens out per sec: %.2f\n", total_gen_tokens / benchmark_time);
    fprintf(stderr, "[RESULT] tokens inout per sec: %.2f\n", (total_input_tokens + total_gen_tokens) / benchmark_time);
    // qps
    fprintf(stderr, "[RESULT] requests per sec: %.2f\n", num_request / benchmark_time);

    
    thread_.join();  // blocks forever

    return 0;
}
