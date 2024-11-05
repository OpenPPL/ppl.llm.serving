# PPL LLM Serving

## Overview

`ppl.llm.serving` is a part of `PPL.LLM` system.

![SYSTEM_OVERVIEW](https://github.com/openppl-public/ppl.nn/blob/master/docs/images/llm-system-overview.png)

**We recommend users who are new to this project to read the [Overview of system](https://github.com/openppl-public/ppl.nn/blob/master/docs/en/llm-system-overview.md).**

`ppl.llm.serving` is a serving based on [ppl.nn](https://github.com/openppl-public/ppl.nn) for various Large Language Models(LLMs). This repository contains a server based on gRPC and inference support for [LLaMA](https://github.com/facebookresearch/llama).

## Prerequisites

* Linux running on x86_64 or arm64 CPUs
* GCC >= 9.4.0
* [CMake](https://cmake.org/download/) >= 3.18
* [Git](https://git-scm.com/downloads) >= 2.7.0
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 11.4. 11.6 recommended. (for CUDA)
* Rust & cargo >= 1.8.0. (for Huggingface Tokenizer)

## PPL Server Quick Start

Here is a brief tutorial, refer to [LLaMA Guide](docs/llama_guide.md) for more details.

* Installing Prerequisites(on Debian or Ubuntu for example)

    ```bash
    apt-get install build-essential cmake git
    ```

* Cloning Source Code

    ```bash
    git clone https://github.com/openppl-public/ppl.llm.serving.git
    ```

* Building from Source

    ```bash
    ./build.sh  -DPPLNN_USE_LLM_CUDA=ON  -DPPLNN_CUDA_ENABLE_NCCL=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'" -DPPL_LLM_ENABLE_GRPC_SERVING=ON
    ```

    NCCL is required if multiple GPU devices are used.

    We support **Sync Decode** feature (mainly for offline_inference), which means model forward and decode in the same thread. To enable this feature, compile with marco `-DPPL_LLM_SERVING_SYNC_DECODE=ON`.

* Exporting Models

    Refer to [ppl.pmx](https://github.com/openppl-public/ppl.pmx) for details.

* Running Server

    ```bash
    ./ppl_llm_server \
        --model-dir /data/model \
        --model-param-path /data/model/params.json \
        --tokenizer-path /data/tokenizer.model \
        --tensor-parallel-size 1 \
        --top-p 0.0 \
        --top-k 1 \
        --max-tokens-scale 0.94 \
        --max-input-tokens-per-request 4096 \
        --max-output-tokens-per-request 4096 \
        --max-total-tokens-per-request 8192 \
        --max-running-batch 1024 \
        --max-tokens-per-step 8192 \
        --host 127.0.0.1 \
        --port 23333 
    ```

    You are expected to give the correct values before running the server.

    - `model-dir`: path of models exported by [ppl.pmx](https://github.com/openppl-public/ppl.pmx).
    - `model-param-path`: params of models. `$model_dir/params.json`.
    - `tokenizer-path`: tokenizer files for `sentencepiece`.

* Running client: send request through gRPC to query the model

    ```bash
    ./ppl-build/client_sample 127.0.0.1:23333
    ```
    See [tools/client_sample.cc](tools/client_sample.cc) for more details.

* Benchmarking

    ```bash
    ./ppl-build/client_qps_measure --target=127.0.0.1:23333 --tokenizer=/path/to/tokenizer/path --dataset=tools/samples_1024.json --request_rate=inf
    ```
    See [tools/client_qps_measure.cc](tools/client_qps_measure.cc) for more details. `--request_rate` is the number of request per second, and value `inf` means send all client request with no interval.

* Running inference offline:

    ```bash
    ./offline_inference \
        --model-dir /data/model \
        --model-param-path /data/model/params.json \
        --tokenizer-path /data/tokenizer.model \
        --tensor-parallel-size 1 \
        --top-p 0.0 \
        --top-k 1 \
        --max-tokens-scale 0.94 \
        --max-input-tokens-per-request 4096 \
        --max-output-tokens-per-request 4096 \
        --max-total-tokens-per-request 8192 \
        --max-running-batch 1024 \
        --max-tokens-per-step 8192 \
        --host 127.0.0.1 \
        --port 23333 
    ```
    See [tools/offline_inference.cc](tools/offline_inference.cc) for more details.

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
