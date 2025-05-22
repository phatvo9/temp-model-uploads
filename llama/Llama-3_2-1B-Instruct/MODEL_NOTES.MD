# Llama-3_2-1B-Instruct

[Model source](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)

Model is serving with `lmdeploy`.

Input data type: `['text']`

# Usage

## Set your PAT
Export your PAT as an environment variable. Then, import and initialize the API Client.

Find your PAT in your security settings.

* Linux/Mac: `export CLARIFAI_PAT="your personal access token"`

* Windows (Powershell): `$env:CLARIFAI_PAT="your personal access token"`

## Running the API with Clarifai's Python SDK


```python
# Please run `pip install -U clarifai` before running this script

from clarifai.client import Model


model = Model(url="https://clarifai.com/phatvo/text-generation-pythonic/models/Llama-3_2-1B-Instruct")
prompt = "What's the future of AI?"

# Clarifai style prediction method
## Stream
generated_text = model.generate(prompt=prompt)
for each in generated_text:
    print(each, end='', flush=True)
## Non stream
generated_text = model.predict(prompt=prompt)
print(generated_text)

# OpenAI completion style method
conversion = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
    # Continue adding messages as the conversation progresses
]
## Stream
stream_generated_text = model.stream_chat(messages=conversion)
for chunk in stream_generated_text:
  # chunk is dict ChatCompletionChunk format
  if chunk["choices"]:
    text = chunk["choices"][0]["delta"].get("content", "")
    print(text, end='', flush=True)

## Non stream
generated_text = model.chat(messages=conversion) # dict of ChatCompletion format
print(generated_text["choices"][0]["message"]["content"])

```

# Server extra args

```
--backend turbomind --cache_max_entry_count 0.9 --tensor_parallel_size 1 --max_prefill_token_num 8192 --dtype auto --quant_policy 0 --max_batch_size 16 --device cuda --server_name 0.0.0.0 --server_port 23333
```