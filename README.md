# MapBench

## 📦 Installation

Create the environment with the provided dependencies:

```
conda env create -f environment.yml
```

## 🚀 Inference with GPT

Run inference using your selected GPT model:

```
python infer_gpt.py \
    --save-dir ./outputs \               # Directory to save results
    --api-key <your-api-key> \           # Your OpenAI API key
    --gpt-model gpt-4o-mini-2024-07-18   # GPT model to use

```

> 💡 Replace <your-api-key> with your actual OpenAI API key.

### 📊 Evaluation

Evaluate the model-generated results:

```
python eval.py \
    --result-file ./outputs/google_map.jsonl \  # Path to the result file
    --log-dir ./logs                            # Directory to save evaluation logs

```

> 📝 The log file will have the same name as the result file.
