# Text or Pixels? It Takes Half

Official codebase for the EMNLP 2025 Findings paper:

**"Text or Pixels? It Takes Half: On the Token Efficiency of Visual Text Inputs in Multimodal LLMs"**

*Yanhong Li\*, Zixuan Lan\*, Jiawei Zhou*  
(\*Equal contribution)

<!-- [![Paper](https://img.shields.io/badge/Paper-EMNLP%202025-blue)](https://github.com/yanhong-lbh/text_or_pixels)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) -->

## Overview

This repository demonstrates a simple yet effective approach to compress textual inputs for large language models by rendering text as images.

![Pipeline](images/vllm_pipeline.png)
*Figure 1: Our text-as-image compression pipeline*

By feeding context as a single image instead of raw text tokens, we achieve:

- **~50% token reduction** without performance loss
- **Up to 45% latency improvement** on larger models
- **Competitive performance** on long-context retrieval and summarization tasks

![Text Token Tolerance](images/text_token_tolerance.png)
*Figure 2: Text token tolerance analysis. The maximum text tokens $m^\star$ that can be preserved without accuracy loss, plotted against the visual tokens $k$ generated from the image. Results show a consistent reduction of roughly $1/2$ in decoder tokens.*


## Installation

```bash
git clone https://github.com/yanhong-lbh/text_or_pixels.git
cd text_or_pixels

pip install -r requirements.txt
```

---

## Quick Start

### 1. Generate Long-Context Data

Use the LM Evaluation Harness to generate RULER NIAH (Needle-in-a-Haystack) tasks:

```bash
pip install lm_eval

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
    --tasks niah_single_1 \
    --device cuda:0 \
    --batch_size 32 \
    --log_samples \
    --output_path data/ruler_niah_single_1_len500 \
    --metadata='{"max_seq_lengths":[500]}'
```

### 2. Convert Text to Images

Transform long text contexts into images with specified dimensions:

```bash
python text_to_image.py \
    --input data/ruler_niah_single_1_len500/samples_niah_single_1_2025-06-27T15-36-26.531929.jsonl \
    --data_dir images/ruler_niah_single_1/600_1000_500 \
    --width 600 \
    --height 1000
```

### 3. Run Evaluations

#### GPT-4o-mini

```bash
export OPENAI_API_KEY='your-api-key-here'

python run_gpt.py \
    --model gpt-4o-mini \
    --image_dir images/ruler_niah_single_1/600_1000_500 \
    --file_path data/ruler_niah_single_1_len500/samples_niah_single_1_2025-06-27T15-36-26.531929.jsonl \
    --num_samples 100 \
    --output gpt4o_mini_results.json
```

#### Qwen2.5-VL

```bash
python run_qwen.py \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --image_dir images/ruler_niah_single_1/600_1000_500 \
    --file_path data/ruler_niah_single_1_len500/samples_niah_single_1_2025-06-27T15-36-26.531929.jsonl \
    --num_samples 100 \
    --output qwen_results.json
```
<!-- 
---

## Key Results

### RULER S-NIAH (Long-Context Retrieval)

| Model | Text Tokens | Visual Tokens | Compression | Accuracy |
|-------|-------------|---------------|-------------|----------|
| GPT-4.1-mini | 1,000 | 442 | 56% | 99% |
| Qwen2.5-VL-72B | 1,000 | 418 | 58% | 97% |

### CNN/DailyMail (Document Summarization)

| Model | Method | Tokens | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|--------|--------|---------|---------|---------|
| GPT-4.1-mini | Text-only | 693 | 23.78 | 8.60 | 16.26 |
| GPT-4.1-mini | Text-as-image | 225 (-67%) | 21.98 | 7.40 | 15.31 |
| Qwen2.5-VL-72B | Text-only | 726 | 25.18 | 9.47 | 17.70 |
| Qwen2.5-VL-72B | Text-as-image | 279 (-62%) | 23.28 | 7.54 | 15.53 |


--- -->

## Contact

- Yanhong Li - [yanhongl@allenai.org](mailto:yanhongl@allenai.org)
- Zixuan Lan - [zixuanlan@uchicago.edu](mailto:zixuanlan@uchicago.edu)
- Jiawei Zhou - [jiawei.zhou.1@stonybrook.edu](mailto:jiawei.zhou.1@stonybrook.edu)
