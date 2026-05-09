# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coconut is the official implementation of "Training Large Language Models to Reason in a Continuous Latent Space" (arXiv:2412.06769). It wraps a HuggingFace causal LM with continuous latent reasoning tokens, enabling multi-pass "thinking" within the latent space.

## Commands

**Setup:**
```bash
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

**Preprocess GSM8K data:**
```bash
bash preprocessing/gsm_icot.bash
```

**Train / evaluate (distributed):**
```bash
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py PATH_TO_ARGS
# e.g.: torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
```

Config is YAML-based. See `args/gsm_coconut.yaml` for a full example. Key flags: `only_eval`, `coconut`, `cot`, `no_thoughts`, `no_cot`, `internalize_cot`.

## Architecture

### Debug: Latent Token Decoding

在 eval 配置中添加 `debug_latent_k: N`，模型会对每个样本的 latent token hidden state 通过 `lm_head` 解码并打印 top-N token 及概率。仅在 `coconut: True` 时生效。

**`coconut.py`** — `forward()` 在每次 feedback 替换时收集 `latent_hidden_states`（batch 0 的 hidden state），通过 `Outputs` namedtuple 返回。`generate()` 新增 `return_latent_hidden` 参数，可选返回 latent hidden states。

**`run.py`** — 新增 `debug_latent_k` 配置（默认 0）。eval 循环中当 `coconut=True` 且 `debug_latent_k > 0` 时，启用 latent 解码并打印所有样本的完整结果。

**`coconut.py`** — The `Coconut` model wraps a base causal LM (GPT2 or Llama). Its `forward()` performs multi-pass inference over sequences containing `<|latent|>` tokens:
- Each pass computes hidden states for a segment, then replaces the next `<|latent|>` token with the preceding hidden state (feedback). KV cache is carried between passes.
- `generate()` does one full forward pass to resolve latent tokens, then autoregressively decodes the answer (batch_size=1 only).

**`run.py`** — Entry point. Handles distributed setup (FSDP for training, DDP for eval), multi-stage curriculum learning, checkpoint save/resume, and the training loop. Training stages progressively replace reasoning steps with latent tokens (controlled by `epochs_per_stage`, `max_latent_stage`, `c_thought`).

**`dataset.py`** — Loads JSON data (`question`/`steps`/`answer`), tokenizes, and constructs input sequences with latent markers. `MyCollator` aligns latent tokens across batch items by left-padding, enabling KV cache reuse. `get_question_latent_dataset` builds eval prompts; `get_cot_latent_dataset` builds training sequences with masked labels.

**`utils.py`** — `Config` (dict-to-attr access) and `set_seed`.

## Key Design Details

- **Multi-stage curriculum**: Stage 0 trains with full CoT. Each subsequent stage replaces one reasoning step with `c_thought` latent tokens. `max_latent_stage` caps the number of replaced steps.
- **Training pipeline for GSM8K**: First train a CoT model (`cot: True`), then initialize Coconut from its checkpoint by setting `load_model_path` and `resume` to skip stage 0.
- **Three special tokens** are added to the tokenizer: `<|start-latent|>`, `<|end-latent|>`, `<|latent|>`. Their embeddings are initialized from the `<<` token.
- **Checkpoint auto-resume**: If checkpoints exist in `save_path`, the run automatically resumes from the latest one, ignoring the `resume` config value.
- Data must be JSON: `[{"question": "...", "answer": "...", "steps": ["...", ...]}, ...]`.
