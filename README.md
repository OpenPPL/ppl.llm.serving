# PPL LLM Serving

## Overview

`ppl.llm.serving` is a serving based on [ppl.nn.llm](https://github.com/openppl-public/ppl.nn.llm) for various Large Language Models(LLMs). This repository contains a server based on gRPC and inference support for [LLaMA](https://github.com/facebookresearch/llama).

## Prerequisites

* Linux running on x86_64 or arm64 CPUs
* GCC >= 9.4.0
* [CMake](https://cmake.org/download/) >= 3.18
* [Git](https://git-scm.com/downloads) >= 2.7.0
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 11.4. 11.6 recommended. (for CUDA)

## Quick Start

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
    ./build.sh -DPPLNN_USE_LLM_CUDA=ON -DPPLNN_CUDA_ENABLE_NCCL=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'"
    ```

    NCCL is required if multiple GPU devices are used.

* Exporting Models

    Refer to [ppl.pmx](https://github.com/openppl-public/ppl.pmx) for details.

* Running Server

    ```bash
    ./ppl-build/ppl_llama_server /path/to/server/config.json
    ```

    Server config examples can be found in `src/models/llama/conf`. You are expected to give the correct values before running the server.

    - `model_dir`: path of models exported by [ppl.pmx](https://github.com/openppl-public/ppl.pmx).
    - `model_param_path`: params of models. `$model_dir/params.json`.
    - `tokenizer_path`: tokenizer files for `sentencepiece`.
    
* Running client: send request through gRPC to query the model

    ```bash
    ./ppl-build/client_sample 127.0.0.1:23333
    ```
    See [tools/client_sample.cc](tools/client_sample.cc) for more details.

* Benchmarking

    ```bash
    ./ppl-build/client_qps_measure 127.0.0.1:23333 /path/to/tokenizer/path tools/samples_1024.json
    ```
    See [tools/client_qps_measure.cc](tools/client_qps_measure.cc) for more details.

Refer to [Documentation](docs/) for more details.

### License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
