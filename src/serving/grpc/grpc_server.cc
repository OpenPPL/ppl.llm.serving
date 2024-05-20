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
#include "utils/mpsc_request_scheduler.h"
#include "utils/profiling_utils.h"

#include "ppl/common/log.h"

#include <list>

using namespace std;
using namespace ppl::common;
using namespace grpc;

namespace ppl { namespace llm {

void GRPCConnection::OnProfiling(const Profiler& profiler) {
    utils::PrintProfiler(profiler);
}

static void AcquireEvent(GRPCEvent* event) {
    event->refcount.fetch_add(1, std::memory_order_acq_rel);
}

static void ReleaseEvent(GRPCEvent* event) {
    uint32_t prev = event->refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        delete event;
    }
}

bool GRPCConnection::AddInfo(uint64_t id, const GRPCReqInfo& info) {
    pthread_mutex_lock(&id2info_lock_);
    auto ret_pair = id2info_.insert(make_pair(id, info));
    bool ok = ret_pair.second;
    pthread_mutex_unlock(&id2info_lock_);
    return ok;
}

// ReleaseEvent() is needed after using info
void GRPCConnection::FindInfo(uint64_t id, GRPCReqInfo* info) {
    pthread_mutex_lock(&id2info_lock_);
    auto ref = id2info_.find(id);
    if (ref != id2info_.end()) {
        *info = ref->second;
        AcquireEvent(ref->second.event);
    }
    pthread_mutex_unlock(&id2info_lock_);
}

void GRPCConnection::RemoveInfo(uint64_t id, GRPCReqInfo* info) {
    pthread_mutex_lock(&id2info_lock_);
    auto ref = id2info_.find(id);
    if (ref != id2info_.end()) {
        *info = ref->second;
        id2info_.erase(ref);
    }
    pthread_mutex_unlock(&id2info_lock_);
}

static void SendBatchRes(proto::BatchedResponse&& pb_batch_res, int nr_finished_req, GRPCEvent* event) {
    pthread_mutex_lock(&event->send_lock);
    event->nr_finished_req += nr_finished_req;
    event->send_queue.emplace_back(std::move(pb_batch_res));
    if (event->send_queue.size() == 1) {
        AcquireEvent(event);
        event->writer.Write(event->send_queue.front(), event);
    }
    pthread_mutex_unlock(&event->send_lock);
}

void GRPCConnection::Send(const vector<Response>& res_list) {
    for (auto& res: res_list) {
        GRPCReqInfo info;
        FindInfo(res.id, &info);
        if (!info.event) {
            // event was removed by some failed response in NewCallThreadFunc()
            continue;
        }

        bool is_last = (res.flag == Response::IS_LAST);
        proto::BatchedResponse pb_batch_res;
        auto* pb_res = pb_batch_res.add_rsp();
        pb_res->set_status(is_last ? proto::FINISHED : proto::PROCESSING);
        pb_res->set_id(info.orig_id);
        if (!res.generated.empty()) {
            pb_res->set_generated(res.generated);
        } else {
            auto* tokens = pb_res->mutable_tokens();
            tokens->add_ids(res.token);
        }

        SendBatchRes(std::move(pb_batch_res), is_last, info.event);

        if (is_last) {
            GRPCReqInfo deleted_info;
            RemoveInfo(res.id, &deleted_info);
            if (deleted_info.event) {
                ReleaseEvent(deleted_info.event); // corresponding to AcquireEvent() before AddInfo()
            }
        }

        ReleaseEvent(info.event); // corresponding to FindInfo()
    }
}

void GRPCConnection::NotifyFailure(uint64_t id) {
    GRPCReqInfo info;
    RemoveInfo(id, &info);
    if (!info.event) {
        return;
    }

    proto::BatchedResponse pb_batch_res;
    auto* pb_res = pb_batch_res.add_rsp();
    pb_res->set_status(proto::FAILED);
    pb_res->set_id(info.orig_id);
    SendBatchRes(move(pb_batch_res), 1, info.event);
    ReleaseEvent(info.event); // corresponding to AcquireEvent() before AddInfo()
}

/* ------------------------------------------------------------------------- */

GRPCServer::GRPCServer(GRPCConnection* c, const function<void(uint64_t)>& on_disconnected_func) {
    arg_.conn = c;
    arg_.on_disconnected_func = on_disconnected_func;
}

void GRPCServer::Loop(RequestProcessor* processor) {
    auto cq = arg_.notification_cq.get();
    auto service = &arg_.service;

    // prepare to process one request
    auto new_event = new GRPCEvent();
    AcquireEvent(new_event);
    service->RequestGeneration(&new_event->ctx, &new_event->pb_req, &new_event->writer, arg_.new_call_cq.get(),
                               arg_.notification_cq.get(), new_event);

    while (true) {
        bool ok;
        void* tag = nullptr;
        if (!cq->Next(&tag, &ok)) {
            LOG(ERROR) << "get request failed. server down.";
            break;
        }

        auto event = static_cast<GRPCEvent*>(tag);
        switch (event->status) {
            case GRPCEvent::NEW: {
                // prepare to process next request
                new_event = new GRPCEvent();
                AcquireEvent(new_event);
                service->RequestGeneration(&new_event->ctx, &new_event->pb_req, &new_event->writer,
                                           arg_.new_call_cq.get(), arg_.notification_cq.get(), new_event);

                if (event->pb_req.req_size() == 0) {
                    ReleaseEvent(event); // corresponding to AcquireEvent() before writer.Write()
                    break;
                }

                // change status to SENDING in case that request(s) are sent before processing is done
                event->status = GRPCEvent::SENDING;

                // generate mapped id first. in case connection is gone before processing is done.
                event->mapped_id_start = uuid_seq_;
                uuid_seq_ += event->pb_req.req_size();

                /*
                  Add event info before processing.
                  If the first response fails, it will remove all infos associated with the same event
                  in NewCallThreadFunc().
                  This can reduce unnecessary work in Send().
                */
                for (int i = 0; i < event->pb_req.req_size(); ++i) {
                    auto& pb_req = event->pb_req.req(i);
                    AcquireEvent(event);
                    arg_.conn->AddInfo(event->mapped_id_start + i, {pb_req.id(), event});
                }

                for (int req_idx = 0; req_idx < event->pb_req.req_size(); ++req_idx) {
                    auto& pb_req = event->pb_req.req(req_idx);
                    auto req = make_shared<Request>();
                    req->id = event->mapped_id_start + req_idx;
                    if (!pb_req.prompt().empty()) {
                        req->prompt = pb_req.prompt();
                    } else {
                        req->token_ids = std::make_shared<std::vector<int>>(pb_req.tokens().ids().begin(), pb_req.tokens().ids().end());
                        req->stop_tokens = std::make_shared<std::unordered_set<int>>(pb_req.stopping_parameters().stop_tokens().ids().begin(), pb_req.stopping_parameters().stop_tokens().ids().begin());
                    }
                    req->temperature = pb_req.temperature();
                    req->generation_length = pb_req.stopping_parameters().max_new_tokens();
                    req->early_stopping = !pb_req.stopping_parameters().ignore_eos_token();
                    processor->Process(req);
                }

                ReleaseEvent(event); // corresponding to AcquireEvent() before RequestGeneration()
                break;
            }
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

        auto event = static_cast<GRPCEvent*>(tag);
        if (!ok) {
            LOG(ERROR) << "client disconnected. waiting for next one...";
            if (tag) {
                // cannot enqueue if the connection is invalid
                if (event->mapped_id_start != UINT64_MAX) {
                    pthread_mutex_lock(&event->send_lock);
                    if (event->mapped_id_start != UINT64_MAX) {
                        uint64_t id_end = event->mapped_id_start + event->pb_req.req_size();
                        for (uint64_t id = event->mapped_id_start; id < id_end; ++id) {
                            GRPCReqInfo info;
                            targ->conn->RemoveInfo(id, &info);
                            if (info.event) {
                                targ->on_disconnected_func(id);
                                ReleaseEvent(info.event); // corresponding to AcquireEvent() before AddInfo()
                            }
                        }
                        event->mapped_id_start = UINT64_MAX;
                    }
                    pthread_mutex_unlock(&event->send_lock);
                }
                ReleaseEvent(event); // corresponding to AcquireEvent() before writer.Write()
            }
            continue;
        }

        switch (event->status) {
            case GRPCEvent::SENDING:
                pthread_mutex_lock(&event->send_lock);
                event->send_queue.pop_front();
                if (event->send_queue.empty()) {
                    if (event->nr_finished_req == event->pb_req.req_size()) {
                        event->status = GRPCEvent::FINISHED;
                        AcquireEvent(event);
                        event->writer.Finish(grpc::Status::OK, event);
                    }
                } else {
                    AcquireEvent(event);
                    event->writer.Write(event->send_queue.front(), event);
                }
                pthread_mutex_unlock(&event->send_lock);
                // fall through
            case GRPCEvent::FINISHED:
                ReleaseEvent(event); // corresponding to AcquireEvent() before writer.Write() or writer.Finish()
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
