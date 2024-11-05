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

#include "resource_manager.h"
#include "post_processor.h"
#include "../../utils/utils.h"

#include "ppl/nn/models/onnx/runtime_builder.h"
#include "ppl/nn/models/onnx/runtime_builder_factory.h"
#include "ppl/common/cuda/cuda_env.h"
#include "ppl/common/cuda/nccl_utils.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/runtime_builder_factory.h"
#include "ppl/nn/models/pmx/load_model_options.h"
#include "ppl/nn/models/pmx/save_model_options.h"
#endif

#include <map>

using namespace ppl::common;
using namespace ppl::nn;
using namespace std;

namespace ppl { namespace llm { namespace cuda {

static const map<int, int32_t> g_cache_int2size = {{0, sizeof(float16_t)}, {8, sizeof(int8_t)}};

static Engine* CreateCudaEngine(int device_id, const ResourceConfig::EngineConfig& engine_config, cudaStream_t stream) {
    ppl::nn::llm::cuda::EngineOptions options;
    options.device_id = device_id;
    options.mm_policy = ppl::nn::llm::cuda::MM_COMPACT;
    options.runtime_stream = stream;

    if (engine_config.quant_method == "none") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_NONE;
    } else if (engine_config.quant_method == "online_i8i8") {
        options.quant_method = ppl::nn::llm::cuda::QUANT_METHOD_ONLINE_I8I8;
    } else {
        LOG(ERROR) << "unknown/unsupported --quant-method option: " << engine_config.quant_method;
        return nullptr;
    }

    if (engine_config.cublas_layout_hint == "default") {
        options.cublas_layout_hint = ppl::nn::llm::cuda::CUBLAS_LAYOUT_DEFAULT;
    } else if (engine_config.cublas_layout_hint == "ampere") {
        options.cublas_layout_hint = ppl::nn::llm::cuda::CUBLAS_LAYOUT_AMPERE;
    } else {
        LOG(ERROR) << "unknown/unsupported --cublas-layout-hint option: " << engine_config.cublas_layout_hint;
        return nullptr;
    }

    auto engine = std::unique_ptr<ppl::nn::Engine>(ppl::nn::llm::cuda::EngineFactory::Create(options));
    if (!engine) {
        LOG(ERROR) << "create cuda engine failed.";
        return nullptr;
    }

    ppl::common::RetCode rc;
    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_SHM_MHA,
                           engine_config.disable_decoding_shm_mha ? 0 : 1);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_SHM_MHA failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_INF_MHA,
                           engine_config.disable_decoding_inf_mha ? 0 : 1);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_INF_MHA failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_INF_GQA,
                           engine_config.disable_decoding_inf_gqa ? 0 : 1);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_INF_GQA failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_ATTN_SPLIT_K,
                           engine_config.configure_decoding_attn_split_k);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_ATTN_SPLIT_K failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_DECODING_ATTN_TPB, engine_config.specify_decoding_attn_tpb);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_DECODING_ATTN_TPB failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_GRAPH_FUSION, engine_config.disable_graph_fusion ? 0 : 1);
    if (ppl::common::RC_SUCCESS != rc) {
        LOG(ERROR) << "configure ENGINE_CONF_GRAPH_FUSION failed: " << ppl::common::GetRetCodeStr(rc);
        return nullptr;
    }

    return engine.release();
}

static Runtime* CreatePPLRuntime(Engine* cuda_engine, const string& model_file) {
    auto builder = unique_ptr<onnx::RuntimeBuilder>(onnx::RuntimeBuilderFactory::Create());
    if (!builder) {
        LOG(ERROR) << "create onnx builder failed.";
        return nullptr;
    }

    auto rc = builder->LoadModel(model_file.c_str());
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "load model [" << model_file << "] failed: " << GetRetCodeStr(rc);
        return nullptr;
    }

    onnx::RuntimeBuilder::Resources resources;
    resources.engines = &cuda_engine;
    resources.engine_num = 1;

    rc = builder->SetResources(resources);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "set resources for builder failed: " << GetRetCodeStr(rc);
        return nullptr;
    }

    rc = builder->Preprocess();
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "builder preprocess failed: " << GetRetCodeStr(rc);
        return nullptr;
    }

    return builder->CreateRuntime();
}

#ifdef PPLNN_ENABLE_PMX_MODEL
static ppl::nn::Runtime* CreatePMXPPLRuntime(ppl::nn::Engine* cuda_engine, const std::string& model_file) {
    auto builder = std::unique_ptr<ppl::nn::pmx::RuntimeBuilder>(ppl::nn::pmx::RuntimeBuilderFactory::Create());
    if (!builder) {
        LOG(ERROR) << "create PmxRuntimeBuilder failed.";
        return nullptr;
    }

    ppl::nn::pmx::RuntimeBuilder::Resources resources;
    resources.engines = &cuda_engine;
    resources.engine_num = 1;

    std::string external_data_dir_fix;
    ppl::nn::pmx::LoadModelOptions opt;
    auto status = builder->LoadModel(model_file.c_str(), resources, opt);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "PmxRuntimeBuilder LoadModel failed: " << ppl::common::GetRetCodeStr(status);
        return nullptr;
    }

    status = builder->Preprocess();
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "pmx preprocess failed: " << ppl::common::GetRetCodeStr(status);
        return nullptr;
    }

    return builder->CreateRuntime();
}
#endif // PPLNN_ENABLE_PMX_MODEL

static void StreamDeleter(void* s) {
    cudaStreamDestroy((cudaStream_t)s);
}

std::unique_ptr<ppl::llm::PostProcessor> CudaResourceManager::CreateCudaPostProcessor(Runtime* runtime) {
    DeviceContext::Type needed_type;
    *((int64_t*)needed_type.str) = 0;
    needed_type.str[0] = 'c';
    needed_type.str[1] = 'u';
    needed_type.str[2] = 'd';
    needed_type.str[3] = 'a';

    DeviceContext* dev = nullptr;
    for (uint32_t i = 0; i < runtime->GetDeviceContextCount(); ++i) {
        if (runtime->GetDeviceContext(i)->GetType() == needed_type) {
            dev = runtime->GetDeviceContext(i);
            break;
        }
    }

    if (!dev) {
        LOG(ERROR) << "cannot find cuda device in runtime.";
        return std::unique_ptr<ppl::llm::PostProcessor>();
    }

    cudaStream_t stream;
    auto rc = dev->Configure(ppl::nn::llm::cuda::DEV_CONF_GET_STREAM, &stream);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "Configure ppl::nn::llm::cuda::DEV_CONF_GET_STREAM failed: " << GetRetCodeStr(rc);
        return std::unique_ptr<ppl::llm::PostProcessor>();
    }
    return std::unique_ptr<ppl::llm::PostProcessor>(new CudaPostProcessor(stream));
}

RetCode InitTask(uint32_t id, const string& model_dir, const std::string& model_format, uint64_t kv_cache_block_bytes,
                 uint64_t kv_scale_block_bytes, float kv_cache_max_tokens_scale, const string& quant_method,
                 int32_t max_running_batch, int32_t vocab_size, bool enable_penalty,
                 const ResourceConfig::EngineConfig& engine_config, Barrier* alloc_max_mem_barrier,
                 CudaResourceManager* mgr) {
    auto rc = InitCudaEnv(id);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "InitCudaEnv for device [" << id << "] failed.";
        return rc;
    }

    cudaStream_t stream;
    auto cu_ret = cudaStreamCreate(&stream);
    if (cu_ret != cudaSuccess) {
        LOG(ERROR) << "cudaStreamCreate failed: " << cudaGetErrorString(cu_ret);
        return RC_DEVICE_RUNTIME_ERROR;
    }
    unique_ptr<void, void (*)(void*)> __stream_guard(stream, StreamDeleter);

    auto engine = unique_ptr<Engine>(CreateCudaEngine(id, engine_config, stream));
    if (!engine) {
        LOG(ERROR) << "create cuda engine [" << id << "] failed.";
        return RC_OTHER_ERROR;
    }

#ifdef PPLNN_CUDA_ENABLE_NCCL
    rc = engine->Configure(ppl::nn::llm::cuda::ENGINE_CONF_SET_TP_NCCL_COMM, mgr->nccl_comm_list[id]);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "engine configure nccl error";
        return RC_OTHER_ERROR;
    }
#endif

    LOG(INFO) << "create engine [" << id << "] success.";

    ppl::nn::llm::cuda::DeviceOptions dev_options;
    dev_options.mm_policy = ppl::nn::llm::cuda::MM_COMPACT;
    dev_options.device_id = id;
    dev_options.stream = stream;

    unique_ptr<ppl::nn::DeviceContext> input_output_device(
        ppl::nn::llm::cuda::EngineFactory::CreateDeviceContext(dev_options));
    if (!input_output_device) {
        LOG(ERROR) << "create device for input/output failed: ";
        return RC_DEVICE_RUNTIME_ERROR;
    }

    unique_ptr<ppl::nn::DeviceContext> host_device(
        ppl::nn::llm::cuda::EngineFactory::CreateHostDeviceContext(ppl::nn::llm::cuda::HostDeviceOptions()));
    if (!host_device) {
        LOG(ERROR) << "create host device failed.";
        return RC_OUT_OF_MEMORY;
    }

    unique_ptr<Runtime> runtime;
    // TODO load models one by one to reduce memory usage
    {
#ifndef PPLNN_ENABLE_PMX_MODEL
        bool use_pmx = model_format == "pmx";
        if (use_pmx) {
            LOG(ERROR) << "enable PPLNN_ENABLE_PMX_MODEL option to use pmx model.";
            return RC_OTHER_ERROR;
        }
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
        if (use_pmx) {
            const string model_path = model_dir + "/model_slice_" + std::to_string(id) + "/model.pmx";
            LOG(INFO) << "model_slice_" << std::to_string(id) << ": " << model_path;
            runtime = unique_ptr<Runtime>(CreatePMXPPLRuntime(engine.get(), model_path));
        } else
#endif
        {
            const string model_path = model_dir + "/model_slice_" + std::to_string(id) + "/model.onnx";

            LOG(INFO) << "model_slice_" << std::to_string(id) << ": " << model_path;
            runtime = unique_ptr<Runtime>(CreatePPLRuntime(engine.get(), model_path));
        }

        if (!runtime) {
            LOG(ERROR) << "create runtime [" << id << "] failed.";
            return RC_OTHER_ERROR;
        }

        for (uint32_t i = 0; i < runtime->GetInputCount(); ++i) {
            auto tensor = runtime->GetInputTensor(i);
            tensor->SetDeviceContext(input_output_device.get());
        }
        for (uint32_t i = 0; i < runtime->GetOutputCount(); ++i) {
            auto tensor = runtime->GetOutputTensor(i);
            tensor->SetDeviceContext(input_output_device.get());
        }
    }

    {
        InferRuntimeParam param;
        param.stream = stream;
        param.engine = std::move(engine);
        param.input_output_device = std::move(input_output_device);
        mgr->runtime_param_list[id] = std::move(param);
    }

    if (id == 0) {
        mgr->post_processor = mgr->CreateCudaPostProcessor(runtime.get());
        if (!mgr->post_processor) {
            LOG(ERROR) << "CreatePostProcessor failed";
            return RC_OTHER_ERROR;
        }

        rc = mgr->post_processor->InitPostProcessorMem(max_running_batch, vocab_size, enable_penalty);
        if (rc != RC_SUCCESS) {
            LOG(ERROR) << "Init Penalty Memory Error";
            return RC_OTHER_ERROR;
        }
    }

    if (id == 0) {
        size_t avail_bytes = 0, total = 0;
        cudaMemGetInfo(&avail_bytes, &total);
        const uint64_t kv_cache_max_bytes = kv_cache_max_tokens_scale * avail_bytes * (kv_cache_block_bytes) /
            (kv_cache_block_bytes + kv_scale_block_bytes);
        const uint64_t kv_scale_max_bytes = kv_cache_max_tokens_scale * avail_bytes * (kv_scale_block_bytes) /
            (kv_cache_block_bytes + kv_scale_block_bytes);
        LOG(INFO) << "avail_bytes: " << avail_bytes;
        LOG(INFO) << "kv_cache_max_bytes: " << kv_cache_max_bytes;
        LOG(INFO) << "kv_scale_max_bytes: " << kv_scale_max_bytes;

        mgr->kv_cache_max_tokens = kv_cache_max_bytes / kv_cache_block_bytes;
        LOG(INFO) << "max_tokens: " << mgr->kv_cache_max_tokens;
    }

    alloc_max_mem_barrier->Wait();

    ResourceItem item;

    cu_ret = cudaMalloc(&item.kv_cache_mem, mgr->kv_cache_max_tokens * kv_cache_block_bytes);
    if (cu_ret != cudaSuccess) {
        LOG(ERROR) << "alloc kv cache [" << mgr->kv_cache_max_tokens * kv_cache_block_bytes
                   << "] failed: " << cudaGetErrorString(cu_ret);
        return RC_OTHER_ERROR;
    }
    if (kv_scale_block_bytes > 0) {
        cu_ret = cudaMalloc(&item.kv_scale_mem, mgr->kv_cache_max_tokens * kv_scale_block_bytes);
        if (cu_ret != cudaSuccess) {
            cudaFree(item.kv_cache_mem);
            LOG(ERROR) << "alloc kv scale [" << mgr->kv_cache_max_tokens * kv_scale_block_bytes
                       << "] failed: " << cudaGetErrorString(cu_ret);
            return RC_OTHER_ERROR;
        }
    }

    item.runtime = runtime.release();
    item.host_device = host_device.release();
    item.engine = mgr->runtime_param_list[id].engine.get();
    mgr->items[id] = item;

    __stream_guard.release();
    return RC_SUCCESS;
}

RetCode CudaResourceManager::Init(const ModelConfig& model_config, const ResourceConfig& resource_config) {
    auto size_iter = g_cache_int2size.find(model_config.cache_quant_bit);
    if (size_iter == g_cache_int2size.end()) {
        LOG(ERROR) << "no supported cache quant bit: [" << model_config.cache_quant_bit << "]";
        return RC_OTHER_ERROR;
    }
    int32_t size_cache_datatype = size_iter->second;

    const uint64_t kv_cache_block_bytes = model_config.num_layers * 2 * model_config.num_kv_heads /
        resource_config.tensor_parallel_size * model_config.hidden_dim / model_config.num_heads * size_cache_datatype;
    uint64_t kv_scale_block_bytes = 0;
    if (model_config.cache_quant_bit > 0) {
        kv_scale_block_bytes = model_config.num_layers * 2 * model_config.num_kv_heads /
            resource_config.tensor_parallel_size * model_config.hidden_dim / model_config.num_heads /
            model_config.cache_quant_group * sizeof(float16_t);
    }
    const int tensor_parallel_size = resource_config.tensor_parallel_size;

    RetCode rc;
#ifdef PPLNN_CUDA_ENABLE_NCCL
    rc = InitNccl(tensor_parallel_size, &nccl_comm_list);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "NCCL init failed.";
        return rc;
    }
    LOG(INFO) << "Init Nccl successed";
#else
    if (tensor_parallel_size > 1) {
        LOG(ERROR)
            << "tensor_parallel_size > 1 need nccl support. Please compile with marco -DPPLNN_CUDA_ENABLE_NCCL=ON";
        return RC_OTHER_ERROR;
    }
#endif

    this->runtime_param_list.resize(tensor_parallel_size);
    this->items.resize(tensor_parallel_size);

    rc = this->device_worker_pool_.Init(tensor_parallel_size);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "init device worker failed.";
        return rc;
    }

    Barrier alloc_max_mem_barrier;
    alloc_max_mem_barrier.Reset(tensor_parallel_size);
    rc = ppl::llm::utils::ParallelExecute(
        InitTask, &this->device_worker_pool_, resource_config.model_dir, resource_config.model_format,
        kv_cache_block_bytes, kv_scale_block_bytes, resource_config.max_tokens_scale,
        resource_config.engine_config.quant_method, resource_config.max_running_batch, model_config.vocab_size,
        resource_config.enable_penalty, resource_config.engine_config, &alloc_max_mem_barrier, this);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "ParallelExecute(InitTask) failed.";
        return rc;
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::llm::cuda