# Guide

Config example:

```
{
    "model_type": "baichuan",
    "model_dir":  "/path/to/model/dir",
    "model_param_path": "/path/to/model/dir/params.json",
    "use_pmx": false,

    "tokenizer_path": "/path/to/tokenizer/tokenizer.model",

    "tensor_parallel_size": 1,

    "top_p": 0.0,
    "top_k": 1,

    "quant_method": "none",

    "max_tokens_scale": 0.94,
    "max_tokens_per_request": 4096,
    "max_running_batch": 1024,
    "max_tokens_per_step": 8192,

    "host": "0.0.0.0",
    "port": 23333
}

```

Detail: see [LLaMA](../llama/README.md)