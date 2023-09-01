#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "llm.grpc.pb.h"

#include <grpc++/grpc++.h>
#include <chrono>

// ABSL_FLAG(std::string, target, "localhost:50051", "Server address");
// ABSL_FLAG(std::string, target, "localhost:50052", "Server address");

using namespace grpc;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using namespace std::chrono;
using namespace ppl::llm;

ABSL_FLAG(std::string, target, "localhost:50052", "Server address");

class GenerationClient {
 public:
  GenerationClient(std::shared_ptr<Channel> channel)
      : stub_(proto::LLMService::NewStub(channel)) {}

 // Assembles the client's payload, sends it and presents the response back
  // from the server.
  int Generation(const std::vector<std::string> prompts, int start_tid=0) {
    // Data we are sending to the server.
    std::cout << "start_tid: " << start_tid << std::endl;;
    ClientContext context;
    proto::BatchedRequest req_list;
    std::unordered_map<int, std::string> rsp_stream_store;
    for(size_t i=0; i<prompts.size(); i++) {
        // request
        auto req = req_list.add_req();
        req->set_id(start_tid + i);
        req->set_prompt(prompts[i]);
        req->set_temperature(0.9);
        req->set_generation_length(15-i);
        // req->set_generation_length(15);
        rsp_stream_store[i]="";
    }
    std::cout << req_list.DebugString() << std::endl;
    // response
    proto::Response rsp;
    std::unique_ptr<ClientReader<proto::Response> > reader(stub_->Generation(&context, req_list));

    std::cout << "stream chat: -----------------" << std::endl;
    // stream chat
    auto start = system_clock::now();
    auto first_fill_time = system_clock::now();
    bool is_first_fill = true;

    auto prev_time = system_clock::now();
    int cnt = 0 ;
    while (reader->Read(&rsp)) {
        if(is_first_fill) {
            first_fill_time = system_clock::now();
            is_first_fill = false;
        }

        auto cur_time = system_clock::now();
        auto iter_duration = duration_cast<std::chrono::milliseconds>(cur_time - prev_time);
        std::cout << cnt <<" iteration time: " << iter_duration.count() << " ms" << std::endl;
        prev_time = cur_time;
        cnt++;

        int tid = rsp.id();
        std::cout  << "id: " << rsp.id()<< std::endl;
        std::cout  << "status: " << rsp.status()<< std::endl;
        std::cout  << "generated: " << rsp.generated()<< std::endl;

        std::string rsp_stream = rsp.generated();
        rsp_stream_store[tid] += (rsp_stream + " ");
    }
    auto end   = system_clock::now();

    std::cout << "answer: -----------------" << std::endl;
    for(auto rsp : rsp_stream_store) {
        std::cout << rsp.second << std::endl;
        std::cout << "--------------------" << std::endl;
    }

    auto first_till_duration = duration_cast<std::chrono::milliseconds>(first_fill_time - start);
    auto duration = duration_cast<std::chrono::milliseconds>(end - start);

    std::cout<<"first fill: "<<first_till_duration.count()<<" ms"<<std::endl;

    std::cout<<"total: "<<duration.count()<<" ms"<<std::endl;

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
    int start_tid = 0;
    if (argc == 3)
        start_tid = std::stoi(std::string(argv[2]));
  // We indicate that the channel isn't authenticated (use of
  // InsecureChannelCredentials()).
//   const std::string target_str = "localhost:2511";
    GenerationClient generator(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
//   const std::string prompt("Building a website can be done in 10 simple steps:");
//   const std::vector<std::string> prompts = {
//     "Building a website can be done in 10 simple steps:",
//     "I believe the meaning of life is"
//   };


    // const std::vector<std::string> prompts = {
    //     "I believe the meaning of life is",
    //     "Simply put, the theory of relativity states that ",
    //     "Building a website can be done in 10 simple steps:\n",
    //     "Tweet: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nTweet: \"My day has been 👍\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet: \"This new music video was incredibile\"\nSentiment:",
    //     "Translate English to French:\n\nsea otter => loutre de mer\n\npeppermint => menthe poivrée\n\nplush girafe => girafe peluche\n\ncheese =>"
    // };

    // 构造多batch prompt
    std::string prompt = "Building a website can be done in 10 simple steps:\n";
    const std::vector<std::string> prompts = {3, prompt};

    // std::string prompt;
    // // int input_token_len = 64;
    // int input_token_len = 128;
    // for(int i=0; i<input_token_len; i++) {
    //     prompt += "x";
    // }
    // const std::vector<std::string> prompts = {8, prompt};


  std::cout << "question: -----------------" << std::endl;
  for (auto& str : prompts)     std::cout << str << std::endl;
  generator.Generation(prompts, start_tid);
  return 0;
}
