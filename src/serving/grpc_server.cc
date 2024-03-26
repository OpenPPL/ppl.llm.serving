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

#include "grpc_server.h"
#include "common/connection.h"
#include "utils/queue_request_scheduler.h"

#include "ppl/common/log.h"

#include <list>

using namespace std;
using namespace ppl::common;
using namespace grpc;

namespace ppl { namespace llm {

struct GRPCConnection final : public Connection {
    GRPCConnection() : writer(&ctx) {
        pthread_mutex_init(&send_lock, nullptr);
    }
    ~GRPCConnection() {
        pthread_mutex_destroy(&send_lock);
    }

    void Send(const Response&) override;
    void NotifyFailure(uint64_t) override;

    enum {
        NEW,
        SENDING,
        FINISHED,
    } status = NEW;
    proto::BatchedRequest pb_req;
    ServerContext ctx;

    pthread_mutex_t send_lock;
    int nr_finished_req = 0;
    std::list<proto::Response> send_queue;
    ServerAsyncWriter<proto::Response> writer;
};

static void SendOneRes(proto::Response&& pb_res, uint32_t nr_finished_req, GRPCConnection* event) {
    pthread_mutex_lock(&event->send_lock);
    event->nr_finished_req += nr_finished_req;
    event->send_queue.emplace_back(std::move(pb_res));
    if (event->send_queue.size() == 1) {
        event->writer.Write(event->send_queue.front(), event);
    }
    pthread_mutex_unlock(&event->send_lock);
}

void GRPCConnection::Send(const Response& res) {
    bool is_last = (res.flag == Response::IS_LAST);

    proto::Response pb_res;
    pb_res.set_status(is_last ? proto::FINISHED : proto::PROCESSING);
    pb_res.set_id(res.id);
    pb_res.set_generated(res.generated);
    SendOneRes(std::move(pb_res), is_last, this);
}

void GRPCConnection::NotifyFailure(uint64_t id) {
    proto::Response pb_res;
    pb_res.set_status(proto::FAILED);
    pb_res.set_id(id);
    SendOneRes(std::move(pb_res), 1, this);
}

/* ------------------------------------------------------------------------- */

GRPCServer::GRPCServer() {
    arg_.on_disconnected_func = [](Connection*) {};
}

void GRPCServer::Loop(RequestProcessor* processor) {
    auto cq = arg_.notification_cq.get();
    auto service = &arg_.service;

    // prepare to process one request
    auto new_event = new GRPCConnection();
    service->RequestGeneration(&new_event->ctx, &new_event->pb_req, &new_event->writer, arg_.new_call_cq.get(),
                               arg_.notification_cq.get(), new_event);

    while (true) {
        bool ok;
        void* tag = nullptr;
        if (!cq->Next(&tag, &ok)) {
            LOG(ERROR) << "get request failed. server down.";
            break;
        }

        auto event = static_cast<GRPCConnection*>(tag);
        switch (event->status) {
            case GRPCConnection::NEW:
                // prepare to process next request
                new_event = new GRPCConnection();
                service->RequestGeneration(&new_event->ctx, &new_event->pb_req, &new_event->writer,
                                           arg_.new_call_cq.get(), arg_.notification_cq.get(), new_event);

                if (event->pb_req.req_size() == 0) {
                    delete event;
                    break;
                }

                // change status to SENDING in case that request(s) are sent before Process() finish
                event->status = GRPCConnection::SENDING;

                for (int req_idx = 0; req_idx < event->pb_req.req_size(); ++req_idx) {
                    auto& pb_req = event->pb_req.req(req_idx);
                    auto req = make_shared<Request>();
                    req->id = pb_req.id();
                    req->prompt = pb_req.prompt();
                    req->temperature = pb_req.temperature();
                    req->generation_length = pb_req.generation_length();
                    req->early_stopping = pb_req.early_stopping();
                    processor->Process(req, event);
                }
                break;
            default:
                LOG(ERROR) << "impossible or invalid status [" << (uint32_t)event->status << "] in Loop().";
                return;
        }
    }
}

void* GRPCServer::NewCallThreadFunc(void* arg) {
    auto targ = static_cast<ThreadArg*>(arg);
    auto cq = targ->new_call_cq.get();

    while (true) {
        bool ok;
        void* tag = nullptr;
        if (!cq->Next(&tag, &ok)) {
            LOG(ERROR) << "get request failed. server down.";
            break;
        }

        auto event = static_cast<GRPCConnection*>(tag);
        if (!ok) {
            LOG(ERROR) << "get request failed. waiting for next one...";
            if (tag) {
                targ->on_disconnected_func(event);
                delete event;
            }
            continue;
        }

        switch (event->status) {
            case GRPCConnection::SENDING:
                pthread_mutex_lock(&event->send_lock);
                event->send_queue.pop_front();
                if (event->send_queue.empty()) {
                    if (event->nr_finished_req == event->pb_req.req_size()) {
                        event->status = GRPCConnection::FINISHED;
                        event->writer.Finish(grpc::Status::OK, event);
                    }
                } else {
                    event->writer.Write(event->send_queue.front(), event);
                }
                pthread_mutex_unlock(&event->send_lock);
                break;
            case GRPCConnection::FINISHED:
                targ->on_disconnected_func(event);
                delete event;
                break;
            default:
                LOG(ERROR) << "impossible or invalid status [" << (uint32_t)event->status << "] in NewCallThreadFunc.";
                break;
        }
    }

    return nullptr;
}

RetCode GRPCServer::Init(const string& addr) {
    builder_.AddListeningPort(addr, InsecureServerCredentials());
    builder_.RegisterService(&arg_.service);
    arg_.notification_cq = builder_.AddCompletionQueue();
    arg_.new_call_cq = builder_.AddCompletionQueue();
    server_ = builder_.BuildAndStart();

    int ret = pthread_create(&new_call_thread_, nullptr, NewCallThreadFunc, &arg_);
    if (ret != 0) {
        LOG(ERROR) << "create new call thread failed.";
        return RC_OTHER_ERROR;
    }
    new_call_thread_created_ = true;

    return RC_SUCCESS;
}

GRPCServer::~GRPCServer() {
    server_->Shutdown();
    arg_.notification_cq->Shutdown();
    arg_.new_call_cq->Shutdown();

    if (new_call_thread_created_) {
        pthread_join(new_call_thread_, nullptr);
    }
}

}} // namespace ppl::llm
