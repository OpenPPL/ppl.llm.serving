#include "llm.grpc.pb.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "grpc++/grpc++.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>

using namespace grpc;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using namespace std::chrono;
using namespace ppl::llm;

ABSL_FLAG(std::string, target, "localhost:50052", "Server address");

class GenerationClient {
public:
    GenerationClient(std::shared_ptr<Channel> channel) : stub_(proto::LLMService::NewStub(channel)) {}

    int Generation(const std::vector<std::vector<int>>& batch_prompt_token_ids) {
        // Data we are sending to the server.
        ClientContext context;
        proto::BatchedRequest req_list;
        std::unordered_map<int, std::vector<int>> rsp_stream_store;
        for (size_t i = 0; i < batch_prompt_token_ids.size(); i++) {
            const auto& prompt_token_ids = batch_prompt_token_ids[i];
            // request
            auto req = req_list.add_req();
            req->set_id(i);
            auto* pb_tokens = req->mutable_tokens();
            for (auto token : prompt_token_ids) {
                pb_tokens->add_ids(token);
            }
            req->set_temperature(1);
            auto* stopping_parameters = req->mutable_stopping_parameters();
            stopping_parameters->set_max_new_tokens(64);
            stopping_parameters->set_ignore_eos_token(false);
            rsp_stream_store[i] = {};
        }
        // response
        proto::BatchedResponse batched_rsp;
        std::unique_ptr<ClientReader<proto::BatchedResponse>> reader(stub_->Generation(&context, req_list));

        // stream chat
        auto start = system_clock::now();
        auto first_fill_time = system_clock::now();
        bool is_first_fill = true;

        while (reader->Read(&batched_rsp)) {
            if (is_first_fill) {
                first_fill_time = system_clock::now();
                is_first_fill = false;
            }

            for (const auto& rsp : batched_rsp.rsp()) {
                int tid = rsp.id();
                int token = rsp.tokens().ids().at(0);
                rsp_stream_store[tid].push_back(token);
            }
        }
        auto end = system_clock::now();

        std::cout << "------------------------------" << std::endl;
        std::cout << "--------- Answer -------------" << std::endl;
        std::cout << "------------------------------" << std::endl;

        for (const auto rsp : rsp_stream_store) {
            for (const auto token : rsp.second) {
                std::cout << token << ", ";
            }
            std::cout << std::endl;
            std::cout << "--------------------" << std::endl;
        }

        auto first_till_duration = duration_cast<std::chrono::milliseconds>(first_fill_time - start);
        auto duration = duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "first fill: " << first_till_duration.count() << " ms" << std::endl;

        std::cout << "total: " << duration.count() << " ms" << std::endl;

        Status status = reader->Finish();
        if (status.ok()) {
            std::cout << "Generation rpc succeeded." << std::endl;
        } else {
            std::cerr << "Generation rpc failed." << std::endl;
            return -1;
        }
        return 0;
    }

private:
    std::unique_ptr<proto::LLMService::Stub> stub_;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " host:port" << std::endl;
        return -1;
    }
    const std::string target_str = argv[1];

    GenerationClient generator(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));

    std::vector<int> token_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<std::vector<int>> prompt_token_ids = {3, token_ids};

    std::cout << "------------------------------" << std::endl;
    std::cout << "--------- Question -------------" << std::endl;
    std::cout << "------------------------------" << std::endl;

    for (auto& token_ids : prompt_token_ids) {
        for (int token : token_ids) {
            std::cout << token << ", ";
        }
        std::cout << std::endl;
    }

    generator.Generation(prompt_token_ids);
    return 0;
}
