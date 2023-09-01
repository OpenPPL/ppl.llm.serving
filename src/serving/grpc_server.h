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

#ifndef __SERVING_GRPC_SERVER_H__
#define __SERVING_GRPC_SERVER_H__

#include "llm.grpc.pb.h"
#include "grpcpp/grpcpp.h"
#include "common/server.h"
#include "ppl/common/retcode.h"
#include <pthread.h>
#include <map>
#include <functional>

namespace ppl { namespace llm {

struct GRPCConnection;

class GRPCServer final : public Server {
public:
    GRPCServer();
    ~GRPCServer();
    ppl::common::RetCode Init(const std::string& addr);
    void Loop(RequestProcessor*) override;

    void SetOnDisconnectedFunc(const std::function<void(Connection*)>& f) {
        arg_.on_disconnected_func = f;
    }

private:
    static void* NewCallThreadFunc(void*);
    static void* NotificationThreadFunc(void*);

public:
    struct ThreadArg final {
        ppl::llm::proto::LLMService::AsyncService service;

        /*
          https://groups.google.com/g/grpc-io/c/V4NAQ77PMEo

          notification_cq gets the tag back indicating a call has started.
          All subsequent operations (reads, writes, etc) on that call report back to new_call_cq.
        */
        std::unique_ptr<grpc::ServerCompletionQueue> notification_cq;
        std::unique_ptr<grpc::ServerCompletionQueue> new_call_cq;

        std::function<void(Connection*)> on_disconnected_func;

        RequestProcessor* processor = nullptr;
    };

private:
    grpc::ServerBuilder builder_;
    std::unique_ptr<grpc::Server> server_;

    bool new_call_thread_created_ = false;
    bool notification_thread_created_ = false;
    pthread_t notification_thread_;
    pthread_t new_call_thread_;
    ThreadArg arg_;
};

}} // namespace ppl::llm

#endif
