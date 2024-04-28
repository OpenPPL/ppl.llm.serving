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

#include "common/processor.h"

#include "llm.grpc.pb.h"
#include "grpcpp/grpcpp.h"
#include "ppl/common/retcode.h"

#include <pthread.h>
#include <map>
#include <functional>
#include <atomic>

namespace ppl { namespace llm {

struct GRPCEvent final {
    GRPCEvent() : writer(&ctx), refcount(0) {
        pthread_mutex_init(&send_lock, nullptr);
    }
    ~GRPCEvent() {
        pthread_mutex_destroy(&send_lock);
    }

    enum {
        NEW,
        SENDING,
        FINISHED,
    } status = NEW;
    proto::BatchedRequest pb_req;
    grpc::ServerContext ctx;

    pthread_mutex_t send_lock;
    uint32_t nr_finished_req = 0;
    std::list<proto::BatchedResponse> send_queue;
    grpc::ServerAsyncWriter<proto::BatchedResponse> writer;

    /* mapped ids of pb_req. used to remove info when the connection is gone */
    uint64_t mapped_id_start = UINT64_MAX;

    /*
      an event may be invalid during processing. for example, client is killed before processing is done.
    */
    std::atomic<uint32_t> refcount;

    GRPCEvent(const GRPCEvent&) = delete;
    GRPCEvent(GRPCEvent&&) = delete;
    void operator=(const GRPCEvent&) = delete;
    void operator=(GRPCEvent&&) = delete;
};

struct GRPCReqInfo final {
    uint64_t orig_id = 0;
    GRPCEvent* event = nullptr;
};

class GRPCConnection final : public Connection {
public:
    GRPCConnection() {
        pthread_mutex_init(&id2info_lock_, nullptr);
    }
    ~GRPCConnection() {
        pthread_mutex_destroy(&id2info_lock_);
    }

    bool AddInfo(uint64_t id, const GRPCReqInfo& info);
    void FindInfo(uint64_t id, GRPCReqInfo* info);
    void RemoveInfo(uint64_t id, GRPCReqInfo* info = nullptr);

    void SetOnDisconnectedFunc(const std::function<void(uint64_t)>& f) {
        on_disconnected_func_ = f;
    }

    void Disconnect(uint64_t id) {
        on_disconnected_func_(id);
    }

    void OnTokenize(uint64_t, const std::vector<int>&) override {}
    void Send(const std::vector<Response>&) override;
    void NotifyFailure(uint64_t) override;

private:
    pthread_mutex_t id2info_lock_;
    std::map<uint64_t, GRPCReqInfo> id2info_;
    std::function<void(uint64_t)> on_disconnected_func_ = {};
};

class GRPCServer final {
public:
    GRPCServer(GRPCConnection*);
    ~GRPCServer();
    ppl::common::RetCode Init(const std::string& addr);
    void Loop(RequestProcessor*);

private:
    static void* NewCallThreadFunc(void*);

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

        GRPCConnection* conn = nullptr;
    };

private:
    grpc::ServerBuilder builder_;
    std::unique_ptr<grpc::Server> server_;

    uint64_t uuid_seq_ = 0;
    bool new_call_thread_created_ = false;
    pthread_t new_call_thread_;
    ThreadArg arg_;

private:
    GRPCServer(const GRPCServer&) = delete;
    void operator=(const GRPCServer&) = delete;
    GRPCServer(GRPCServer&&) = delete;
    void operator=(GRPCServer&&) = delete;
};

}} // namespace ppl::llm

#endif
