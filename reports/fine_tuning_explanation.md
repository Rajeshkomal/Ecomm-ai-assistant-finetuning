# Fine-Tuning Explanation

This document explains the concepts and the exact configuration used to build the
E-commerce Database Assistant.

## Why full fine-tuning is expensive

Full fine-tuning updates **every weight** in the model. For even a 1.5B-parameter model
this means storing the weights, their gradients, and the optimizer state (Adam keeps two
extra values per parameter) in GPU memory at once. That is roughly 12-16 bytes per
parameter, so a "small" model can still need tens of GB of VRAM and long training times.
It also produces a full-size copy of the model per task, which is wasteful to store and serve.

## What LoRA does

LoRA (Low-Rank Adaptation) freezes the original model weights and injects small trainable
"adapter" matrices into the attention and MLP layers. Instead of updating a large weight
matrix W, it learns two tiny matrices A and B whose product (a low-rank update) is added to W.
Only these small matrices are trained, so the number of trainable parameters drops by
orders of magnitude, memory use falls, and each fine-tuned task is just a few MB of adapter
weights on top of the shared base model.

## What QLoRA does

QLoRA = **Quantized LoRA**. The frozen base model is loaded in 4-bit precision (NF4)
instead of 16-bit, which cuts the memory needed to hold the model by about 4x. LoRA adapters
are then trained on top of this quantized base in higher precision. The result is almost the
same quality as LoRA but with a fraction of the memory.

## Why QLoRA is useful on a limited GPU

A free Colab T4 has ~15 GB of VRAM. Loading a 1.5B model in 16-bit plus optimizer state for
full fine-tuning would not fit comfortably. With QLoRA the base model occupies ~1-2 GB in
4-bit, leaving room for activations and the small adapter optimizer state, so the whole
three-stage pipeline runs on a single free GPU.

## What is non-instruction fine-tuning?

Non-instruction fine-tuning (also called continued pretraining) trains the model on **raw
domain text** with a plain next-token objective - no question/answer format. Its purpose is
to teach the model the vocabulary, facts, and writing style of the domain. Here it learns the
client schema: table names like `X_Product_Images`, their columns, and how tables relate.

## What is instruction fine-tuning?

Instruction fine-tuning (Supervised Fine-Tuning, SFT) trains the model on
**question -> answer** pairs so it learns to follow instructions. In this project it maps a
natural-language question either to the correct table name (schema discovery) or to a valid
SQL query (support queries).

## What is DPO?

DPO (Direct Preference Optimization) is an alignment method. For each prompt it is given a
**chosen** (better) answer and a **rejected** (worse) answer, and it directly optimizes the
model to raise the probability of the chosen answer relative to the rejected one - without
training a separate reward model as classic RLHF requires. Here it teaches the model to
prefer correct, specific SQL/tables over hallucinated tables, missing `DISTINCT`, lazy
`SELECT *`, and wrong joins.

## Difference between SFT and DPO

- **SFT** learns from *positive examples only* - it imitates the single correct answer for
  each prompt. It teaches the model *what a good answer looks like*.
- **DPO** learns from *pairs* - a better and a worse answer for the same prompt. It teaches
  the model *to prefer good over bad*, sharpening quality, safety, and consistency after SFT.
- Typical order: pretrain -> SFT -> DPO. SFT gives the capability; DPO refines the preference.

## Hyperparameters used

| Parameter | Stage 1 (Non-instruct) | Stage 2 (SFT) | Stage 3 (DPO) |
|-----------|------------------------|---------------|---------------|
| Base model | Qwen2.5-Coder-1.5B | Qwen2.5-Coder-1.5B | Qwen2.5-Coder-1.5B (SFT adapter) |
| Quantization | 4-bit (QLoRA) | 4-bit (QLoRA) | 4-bit (QLoRA) |
| LoRA rank (r) | 16 | 16 | 16 |
| LoRA alpha | 16 | 16 | 16 |
| LoRA dropout | 0 | 0 | 0 |
| Target modules | attn + MLP + embed_tokens + lm_head | attn + MLP | attn + MLP |
| Learning rate | 5e-5 (embed 1e-5) | 2e-4 | 5e-5 |
| Batch size | 2 | 2 | 2 |
| Grad accumulation | 4 (eff. batch 8) | 4 (eff. batch 8) | 4 (eff. batch 8) |
| Epochs | 3 | 3 | 2 |
| Max seq length | 2048 | 2048 | 2048 |
| Optimizer | adamw_8bit | adamw_8bit | adamw_8bit |
| DPO beta | - | - | 0.1 |

**Notes on choices:**
- `r = alpha = 16` is a balanced default; the update scale (alpha/r) is 1.0.
- `dropout = 0` is the Unsloth-optimized setting and trains slightly faster.
- Stage 1 uses a lower LR and also adapts `embed_tokens`/`lm_head` to absorb domain tokens.
- SFT uses the highest LR (2e-4) because it is the main capability-learning stage.
- DPO uses a lower LR (5e-5) and `beta=0.1` to refine preferences without destabilizing SFT.
