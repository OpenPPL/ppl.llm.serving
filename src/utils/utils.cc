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

#include "utils.h"
#include "ppl/nn/runtime/tensor.h"
#include "ppl/common/log.h"
#include "ppl/common/types.h"
#include <fstream>

using namespace ppl::common;
using namespace ppl::nn;
using namespace std;

namespace ppl { namespace llm { namespace utils {

static const char* MemMem(const char* haystack, unsigned int haystack_len, const char* needle,
                          unsigned int needle_len) {
    if (!haystack || haystack_len == 0 || !needle || needle_len == 0) {
        return nullptr;
    }

    for (auto h = haystack; haystack_len >= needle_len; ++h, --haystack_len) {
        if (memcmp(h, needle, needle_len) == 0) {
            return h;
        }
    }
    return nullptr;
}
static void SplitString(const char* str, unsigned int len, const char* delim, unsigned int delim_len,
                        const std::function<bool(const char* s, unsigned int l)>& f) {
    const char* end = str + len;

    while (str < end) {
        auto cursor = MemMem(str, len, delim, delim_len);
        if (!cursor) {
            f(str, end - str);
            return;
        }

        if (!f(str, cursor - str)) {
            return;
        }

        cursor += delim_len;
        str = cursor;
        len = end - cursor;
    }

    f("", 0); // the last empty field
}

void ParseTokens(const std::string& stop_tokens_str, std::set<int>* stop_tokens) {
    SplitString(stop_tokens_str.data(), stop_tokens_str.size(), ",", 1,
                [stop_tokens](const char* s, unsigned int l) -> bool {
                    if (l > 0) {
                        stop_tokens->insert(std::atoi(s));
                    }
                    return true;
                });
    return;
}

// ref: https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector 
uint64_t HashStd(uint64_t prev, const int32_t* vec, int32_t len) {
    uint64_t ret = len;
    ret ^= std::hash<uint64_t>()(prev);
    for (int i=0; i<len; ++i) {
        ret ^= std::hash<uint64_t>()((uint64_t)vec[i]);
    }
    return ret;
}

uint64_t HashCombine(uint64_t prev, const int32_t* vec, int32_t len) {
    uint64_t seed = len;
    seed ^= prev + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    for (int i=0; i<len; ++i) {
        seed ^= vec[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

static const pair<std::string, datatype_t> g_str2datatype[] = {
    {"fp64", DATATYPE_FLOAT64}, {"fp32", DATATYPE_FLOAT32}, {"fp16", DATATYPE_FLOAT16}, {"int32", DATATYPE_INT32},
    {"int64", DATATYPE_INT64},  {"int8", DATATYPE_INT8},    {"bool", DATATYPE_BOOL},    {"", DATATYPE_UNKNOWN},
};

static const char* FindDataTypeStr(datatype_t dt) {
    for (int i = 0; !g_str2datatype[i].first.empty(); ++i) {
        if (g_str2datatype[i].second == dt) {
            return g_str2datatype[i].first.c_str();
        }
    }
    return nullptr;
}

static string GetDimsStr(const Tensor* tensor) {
    auto shape = tensor->GetShape();
    if (shape->GetRealDimCount() == 0) {
        return string();
    }

    string res = ToString(shape->GetDim(0));
    for (uint32_t i = 1; i < shape->GetDimCount(); ++i) {
        res += "_" + ToString(shape->GetDim(i));
    }

    return res;
}

bool SaveInputsOneByOne(const ppl::nn::Runtime* runtime, const std::string& save_dir, const std::string& tag = "") {
    for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
        auto t = runtime->GetInputTensor(c);
        auto shape = t->GetShape();

        auto bytes = shape->CalcBytesIncludingPadding();
        vector<char> buffer(bytes);

        ppl::nn::TensorShape src_desc = *t->GetShape();
        src_desc.SetDataFormat(DATAFORMAT_NDARRAY);
        auto status = t->ConvertToHost(buffer.data(), src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data failed: " << GetRetCodeStr(status);
            return false;
        }

        const char* data_type_str = FindDataTypeStr(shape->GetDataType());
        if (!data_type_str) {
            LOG(ERROR) << "unsupported data type[" << GetDataTypeStr(shape->GetDataType()) << "]";
            return false;
        }

        char name_prefix[32];
        if (tag.empty())
            sprintf(name_prefix, "pplnn_input_%05u_", c);
        else
            sprintf(name_prefix, "pplnn_input_%s_%05u_", tag.c_str(), c);
        const string in_file_name = save_dir + "/" + string(name_prefix) + t->GetName() + "-" +
            GetDimsStr(t) + "-" + string(data_type_str) + ".dat";
        ofstream ofs(in_file_name, ios_base::out | ios_base::binary | ios_base::trunc);
        if (!ofs.is_open()) {
            LOG(ERROR) << "save input file[" << in_file_name << "] failed.";
            return false;
        }

        ofs.write(buffer.data(), bytes);
    }

    return true;
}

}}} // namespace ppl::llm::utils