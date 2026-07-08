# E-commerce Database Assistant — Domain-Specific LLM Fine-Tuning

A domain-specific AI assistant built by fine-tuning an open-source LLM with **Unsloth**,
through three stages: **non-instruction fine-tuning → instruction fine-tuning (SFT) → DPO
preference alignment**.

## Project title

**E-commerce Database Assistant** — an internal assistant for product-support engineers and
new developers working on the client's e-commerce platform.

## Domain selected

**E-commerce Product Support Assistant** (schema/DB assistant flavor).

## Business problem

New developers and support engineers need to understand the client's database quickly. The
assistant answers two kinds of questions about the client schema:

1. **Schema discovery** — e.g. *"Which table stores product images?"* → `X_Product_Images`
   (with key columns).
2. **SQL generation** — e.g. *"Give me a query to find unique orders for a customer"* →
   a valid SQL query against the schema.

This reduces onboarding time and the need to constantly ask senior engineers where data lives.

## Dataset details

All datasets are custom-built around a 30-table, `X_`-prefixed e-commerce schema (catalog,
customer, cart/order, fulfillment, payment/promotion, store).

| Dataset | File | Size | Format |
|---------|------|------|--------|
| Non-instruction (raw corpus) | `data/non_instruction_data.txt` | ~62 paragraphs, 30 tables | plain text (descriptions + DDL + relationships) |
| Instruction | `data/instruction_dataset.jsonl` | 215 examples | `{instruction, response}` |
| Preference | `data/preference_dataset.jsonl` | 63 examples | `{prompt, chosen, rejected}` |

Datasets are also published on the Hugging Face Hub and loaded directly in the notebooks:

- `Rajesh507/ecomm-db-noninstruct`
- `Rajesh507/ecomm-db-instruction`
- `Rajesh507/ecomm-db-preference`

Push them yourself with:

```bash
pip install datasets huggingface_hub
huggingface-cli login
python scripts/push_to_hf.py --user Rajesh507
```

## Base model used

**`unsloth/Qwen2.5-Coder-1.5B`** — a small, GPU-friendly model that is strong at SQL.
(Swap to `unsloth/Qwen2.5-1.5B` by changing one line in the notebooks.)

## Approach

### Non-instruction fine-tuning
Continued pretraining on the raw schema corpus so the model learns the domain vocabulary
(table names, columns, relationships). Trains attention + MLP + `embed_tokens`/`lm_head`.

### Instruction fine-tuning (SFT)
Supervised fine-tuning on 215 question→answer pairs so the model maps a natural-language
question to the correct table name or a valid SQL query. Uses one consistent prompt template.

### DPO alignment
Direct Preference Optimization on 63 preference pairs so the model prefers correct, specific
answers over wrong/generic ones (hallucinated tables, missing `DISTINCT`, lazy `SELECT *`,
wrong joins). Uses `ref_model=None` to fit a free T4 GPU.

## LoRA / QLoRA configuration

| Setting | Value |
|--------|-------|
| Method | QLoRA (4-bit base + LoRA adapters) |
| Rank (r) | 16 |
| Alpha | 16 |
| Dropout | 0 |
| LR (Stage 1 / SFT / DPO) | 5e-5 / 2e-4 / 5e-5 |
| Batch size × grad accum | 2 × 4 (effective 8) |
| Max seq length | 2048 |
| DPO beta | 0.1 |

Full rationale in [`reports/fine_tuning_explanation.md`](reports/fine_tuning_explanation.md).

## How to run (Google Colab)

Datasets load from Hugging Face; trained adapters save to Google Drive.

1. Push datasets to HF: `python scripts/push_to_hf.py --user Rajesh507`
2. Open each notebook in Colab (GPU: T4) and run top to bottom, **in order**:

| Notebook | Stage | Colab |
|----------|-------|-------|
| `notebooks/non_instruction_finetuning.ipynb` | 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rajeshkomal/Ecomm-ai-assistant-finetuning/blob/main/notebooks/non_instruction_finetuning.ipynb) |
| `notebooks/instruction_finetuning.ipynb` | 2 (SFT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rajeshkomal/Ecomm-ai-assistant-finetuning/blob/main/notebooks/instruction_finetuning.ipynb) |
| `notebooks/dpo_alignment.ipynb` | 3 (DPO) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rajeshkomal/Ecomm-ai-assistant-finetuning/blob/main/notebooks/dpo_alignment.ipynb) |

Restart the runtime between notebooks to free GPU memory.

## Inference

```bash
python src/inference.py --adapter /path/to/stage3_dpo_adapter \
    --question "Which table stores product images?"
```

## Repository structure

```
Ecomm-ai-assistant-finetuning/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   └── inference.py
├── scripts/
│   ├── generate_notebooks.py
│   └── push_to_hf.py
├── README.md
└── requirements.txt
```

## Training screenshots or logs

_Add Colab training-loss screenshots or the trainer log output here after running._

## Before vs after output comparison

See [`reports/base_model_evaluation.md`](reports/base_model_evaluation.md),
[`reports/sft_model_comparison.md`](reports/sft_model_comparison.md), and
[`reports/final_evaluation.md`](reports/final_evaluation.md).

## Final observations

- The **base → SFT** step gives the largest jump: from generic/hallucinated answers to
  correct client-specific tables and valid SQL.
- **DPO** refines quality: fewer hallucinated columns, consistent `DISTINCT`, less `SELECT *`.
- Fine-tuning a small model to *recall* an exact schema is the hard part; in production this
  would be complemented with retrieval (RAG) over the schema.

## Challenges faced

- Keeping table/column naming perfectly consistent across all three datasets to avoid
  teaching the model to hallucinate.
- Fitting a three-stage pipeline into a single free T4 GPU (solved with QLoRA and
  `ref_model=None` for DPO).
- Small models are weak at exact factual recall, so schema-discovery questions are harder
  than SQL generation where schema can be supplied in the prompt.

## Future improvements

- Add a RAG layer over the schema for reliable table/column lookup.
- Expand the instruction set (more phrasings per table) and the preference set.
- Execute generated SQL against a real/sample DB for automatic correctness scoring.
- Try a larger model (e.g. 3B) or blend a public text-to-SQL dataset for general robustness.

## Tech stack

Unsloth, TRL, PEFT, Transformers, Datasets, PyTorch, bitsandbytes. See `requirements.txt`.
