## LLaMA Guide 

1. Download model

Download llama model and tokenizer following the documents provided by facebook:
[LLaMA](https://github.com/facebookresearch/llama/tree/llama_v1#llama).

Here we save downloaded file in folder `/model_data/llama_fb/`.

2. Convert model.

Convert through our pmx, following [guide](https://github.com/openppl-public/ppl.pmx/blob/master/model_zoo/llama/facebook/README.md). Here we use llama_7b for example, exporting model in `/model_data/llama_fb/7B/`.

    ```bash
    git clone https://github.com/openppl-public/ppl.pmx.git
    cd ppl.pmx/model_zoo/llama/facebook
    pip install -r requirements.txt # requirements
    MP=1
    OMP_NUM_THREADS=${MP} torchrun --nproc_per_node ${MP} \
    Export.py --ckpt_dir /model_data/llama_fb/7B/ \
    --tokenizer_path /model_data/llama_fb/tokenizer.model \
    --export_path /model_data/llama_7b_ppl/ \
    --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 \
    --quantized_cache 1 --dynamic_batching 1 
    ```
Differenct model require different MP values, for llama_7b `MP=1`.
| MP                   | value |
|----------------------|-------|
| LLaMA-7B             |   1   |
| LLaMA-13B            |   2   |
| LLaMA-30B            |   4   |
| LLaMA-65B            |   8   |

Here, we generate `model_slice_0` and `params.json` in `llama_ppl_7B/`. 
* Folder `model_slice_0` include tensor parallel slice weight and structure in onnx format of llama_7b model. The number of slice is equal to used GPU numbers. For example, llama_13b has two slice folder `model_slice_0` and `model_slice_1`. 
* File `params.json` describe the model llama_7b config, which is differenct with llama_13b and llama_65b.

3. Build from source

```bash
cd /xx/ppl.llm.serving
./build.sh -DPPLNN_USE_LLM_CUDA=ON -DPPLNN_CUDA_ENABLE_NCCL=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'"
```

4. Set server configuration. 

Here we set server configuration in file `src/models/llama/conf/llama_7b_config_example.json`. 

```
{
    "model_type": "llama",
    "model_dir":  "/model_data/llama_7b_ppl/",
    "model_param_path": "/model_data/llama_7b_ppl/params.json",

    "tokenizer_path": "/model_data/llama_fb/tokenizer.model",

    "tensor_parallel_size": 1,

    "top_p": 0.0,
    "top_k": 1,

    "max_tokens_scale": 0.94,
    "max_tokens_per_request": 4096,
    "max_running_batch": 1024,
    "max_tokens_per_step": 8192,

    "host": "0.0.0.0",
    "port": 23333
}

```

where params `model_dir`, `model_param_path` and `tokenizer_path` is from step 1 and step 2, `tensor_parallel_size` is 1 in llama_7b, and would be different in other llama model.

| tensor_parallel_size | value |
|----------------------|-------|
| LLaMA-7B             |   1   |
| LLaMA-13B            |   2   |
| LLaMA-30B            |   4   |
| LLaMA-65B            |   8   |


5. Launch server

Launch server with configuration file in step 4.
```bash
./ppl-build/ppl_llama_server src/models/llama/conf/llama_7b_config_example.json
```

6. Launch client

Send request through [gRPC](https://github.com/grpc/grpc) to query the model.

```bash
./client_sample 127.0.0.1:23333
```
The prompt is writing in source file `tools/client_sample.cc`. If you want to change prompts, you should revise the source file and rebuild it.