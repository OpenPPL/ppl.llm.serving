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

#ifndef __PPL_LLM_INDEX_MANAGER_H__
#define __PPL_LLM_INDEX_MANAGER_H__

#include "ppl/common/compact_addr_manager.h"

namespace ppl { namespace llm {

class IndexManager final {
private:
    class IndexAllocator final : public ppl::common::CompactAddrManager::VMAllocator {
    public:
        void Init(uint64_t max_index) {
            max_ = max_index;
        }
        uintptr_t GetReservedBase() const override {
            return 0;
        }
        uint64_t GetAllocatedSize() const override {
            return used_;
        }
        uint64_t Extend(uint64_t needed) override {
            if (needed + used_ > max_) {
                return 0;
            }

            used_ += needed;
            return needed;
        }

    private:
        uint64_t max_ = 0;
        uint64_t used_ = 0;
    };

public:
    IndexManager() : mgr_(&vmr_) {}
    void Init(uint64_t max_index) {
        nr_avail_blk_ = max_index;
        vmr_.Init(max_index);
    }
    int64_t GetAvailableBlockNum() const {
        return nr_avail_blk_;
    }
    int64_t Alloc(uint64_t nr) {
        auto ret = mgr_.Alloc(nr);
        if (ret == UINTPTR_MAX) {
            return INT64_MAX;
        }
        nr_avail_blk_ -= nr;
        return (int64_t)ret;
    }
    void Free(uint64_t start, uint64_t nr) {
        mgr_.Free(start, nr);
        nr_avail_blk_ += nr;
    }

private:
    uint64_t nr_avail_blk_;
    IndexAllocator vmr_;
    ppl::common::CompactAddrManager mgr_;
};

}} // namespace ppl::llm

#endif
