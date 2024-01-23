# Guide
Here we will explain the config json file with example llama_7b. Presume we've aleady getting exported pmx model (refer to  [ppl.pmx/model_zoo/llama/facebook/README.md](https://github.com/openppl-public/ppl.pmx/blob/master/model_zoo/llama/facebook/README.md) for more detail), saved it in `/model_data/llama_7b_ppl`, and tokenizer saved in `/model_data/llama_fb/tokenizer.model`.

```
{
    "model_type": "llama",
    "model_dir":  "/model_data/llama_7b_ppl",
    "model_param_path": "/model_data/llama_7b_ppl/params.json",
    "use_pmx": false,

    "tokenizer_path": "/model_data/llama_fb/tokenizer.model",

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

- `model_type`: basic type of model, including diffenent weight. For example, model type `llama` including llama-7b, llama-13b, llama-30b and llama-65b. For other llm model, we will support soon.

- `model_dir`: path to the model. 

- `model_param_path`: path to model params.

- `use_pmx`: if use pmx model.

- `tokenizer_path`: path to tokenizer.

- `tensor_parallel_size`: size of tensor parallel. For llama_7b, the value is 1. 

- `top_p`:  select enough tokens to "cover" a certain amount of probability defined by the parameter p (refer to [Token selection strategies: Top-K, Top-p, and Temperature](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/) for more detail). The value range from 0~1.

- `top_k`: select the first top k tokens to create new distribution (refer to [Token selection strategies: Top-K, Top-p, and Temperature](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/) for more detail). This value does not matter on current version. 

- `quant_method`: only accept two value: {`none`, `online_i8i8`}. `none` means not quantize, and `online_i8i8` (also called `w8a8`) means weight and tensor are both quantized with int8. 

- `max_tokens_scale`: Pre-allocated kv cache memory on percentage of freed memory. The value range from 0~1. For example after loaded model weight, there is only 20G freed GPU memory. When we set `max_tokens_scale=0.94`, the pre-allocated kv cache memory is `20G*0.94=18.8G`

- `max_tokens_per_request`: max size of "prompt tokens + generate tokens".

- `max_running_batch`: max size of running batch.

- `max_tokens_per_step`: max input token per step. For the task in prefill stage, the value is prompt token numbers, and for decode stage, the number is always 1. 

- `host`: server ip address.

- `port`: server port.
