"""Generates the three Colab-ready fine-tuning notebooks as valid .ipynb files.
Datasets are loaded from the Hugging Face Hub; trained adapters are pushed to the Hugging Face Hub.
Run: python scripts/generate_notebooks.py
"""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(HERE, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name, cells):
    path = os.path.join(NB_DIR, name)
    with open(path, "w") as f:
        json.dump(notebook(cells), f, indent=1)
    print("wrote", path)


# Shared snippets -----------------------------------------------------------
INSTALL = """# Install Unsloth (needs a CUDA GPU). If Colab prompts to restart after install, do it.
%%capture
!pip install --upgrade unsloth unsloth_zoo
# Ensure a MODERN TRL (provides SFTConfig/DPOConfig + processing_class) that matches new transformers.
!pip install --upgrade "trl>=0.13.1\""""

GPU_CHECK = """import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - set Runtime > Change runtime type > T4 GPU")"""

HF_CFG = """# --- Datasets live on the Hugging Face Hub (pushed via scripts/push_to_hf.py) ---
HF_USER = "Rajesh507"   # <-- your Hugging Face username
DS_NONINSTRUCT = f"{HF_USER}/ecomm-db-noninstruct"
DS_INSTRUCTION = f"{HF_USER}/ecomm-db-instruction"
DS_PREFERENCE  = f"{HF_USER}/ecomm-db-preference"

# If the datasets are PRIVATE, log in first (needs a read token):
# from huggingface_hub import login; login()"""

HF_SAVE = """# Persist trained adapters to the Hugging Face Hub (no Google Drive needed).
from huggingface_hub import login
login()   # paste a WRITE token: https://huggingface.co/settings/tokens

ADAPTER_STAGE1 = f"{HF_USER}/ecomm-db-stage1-noninstruct"
ADAPTER_STAGE2 = f"{HF_USER}/ecomm-db-stage2-sft"
ADAPTER_STAGE3 = f"{HF_USER}/ecomm-db-stage3-dpo"
print("Adapters will be pushed under:", HF_USER)"""

PROMPT_TEMPLATE = '''# A single, consistent prompt template used across ALL three stages.
PROMPT = """Below is a question about the client e-commerce database schema. \\
Write a response that correctly answers it, giving the exact table name(s) or a valid SQL query.

### Question:
{}

### Answer:
{}"""'''

MODEL_CFG = """MODEL_NAME = "unsloth/Qwen2.5-Coder-1.5B"   # Coder variant is stronger at SQL. Alt: "unsloth/Qwen2.5-1.5B"
MAX_SEQ_LEN = 2048"""

LORA = """from unsloth import FastLanguageModel
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                       # LoRA rank
    lora_alpha = 16,              # scaling
    lora_dropout = 0,             # 0 is optimized in Unsloth
    bias = "none",
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)"""

# Stage 1 also adapts the input/output embeddings so the model can absorb new
# X_ schema vocabulary during continued pretraining.
LORA_STAGE1 = """from unsloth import FastLanguageModel
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                       # LoRA rank
    lora_alpha = 16,              # scaling
    lora_dropout = 0,             # 0 is optimized in Unsloth
    bias = "none",
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj",
                      "embed_tokens","lm_head"],   # continued pretraining: also adapt the
                                                   # input/output embeddings so the model can
                                                   # actually absorb new X_ schema vocabulary
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)"""

# Stage 2 config: continue from the Stage-1 domain-adapted adapter.
START_MODEL_CFG = MODEL_CFG + """

# Chained pipeline (bootcamp Class-22): Stage 2 CONTINUES from the Stage-1
# domain-adapted adapter, so the model has already learned the X_ schema
# vocabulary before we teach Q->A behavior here.
# Set to MODEL_NAME only if you deliberately want to skip Stage 1 (not recommended).
START_MODEL = ADAPTER_STAGE1   # fallback: MODEL_NAME"""

# Stage 2 self-check: verbatim training questions must be reproduced, else the
# adapter is not active / undertrained.
SELF_CHECK = """# --- Self-check: is the trained adapter ACTUALLY affecting the output? ---
# These questions are copied VERBATIM from the training set, so a correctly
# trained + loaded adapter must reproduce the domain-specific answer under
# greedy decoding. If this FAILS (generic Orders/shipments tables), the adapter
# is not active or training did not converge - fix that before trusting any
# other output from this model.
checks = [
    ("Give me a query to find unique orders for a customer.", ["X_Order", "DISTINCT", "order_id"]),
    ("Which table stores product images?",                    ["X_Product_Images"]),
    ("Where is shipment and tracking information stored?",    ["X_Shipment"]),
]

all_ok = True
for q, must_have in checks:
    ans = ask(q)
    ok = all(tok.lower() in ans.lower() for tok in must_have)
    all_ok = all_ok and ok
    print(f"[{'PASS' if ok else 'FAIL'}] {q}")
    print("   ->", ans.replace(chr(10), ' ')[:200])
    if not ok:
        print("   expected to contain:", must_have)

print("\\nADAPTER ACTIVE + LEARNED:", all_ok)
assert all_ok, (
    "Verbatim training examples were NOT reproduced. The adapter is not applied "
    "or the model is undertrained. Check: (1) you ran the training cell in THIS "
    "session, (2) START_MODEL points at the Stage-1 adapter, (3) epochs are high "
    "enough, and re-run without restarting the runtime mid-way."
)"""

# ---------------------------------------------------------------------------
# Notebook 1: Non-instruction fine-tuning (continued pretraining on raw text)
# ---------------------------------------------------------------------------
nb1 = [
    md("""# Stage 1 - Non-Instruction Fine-Tuning (Continued Pretraining)

**Goal:** teach the base model the *language and facts* of the client e-commerce schema
by continuing pretraining on the raw schema corpus.

This adapts the model to domain vocabulary (table names like `X_Product_Images`, columns,
relationships) **before** we teach it to follow instructions in Stage 2.

Pipeline: **Base -> [Stage 1] -> Stage 2 (SFT) -> Stage 3 (DPO)**

Data and adapters both live on the Hugging Face Hub."""),
    code(INSTALL),
    code(GPU_CHECK),
    md("## 1. Config + load the raw domain corpus from Hugging Face"),
    code(HF_CFG),
    code(HF_SAVE),
    code("""from datasets import load_dataset
raw = load_dataset(DS_NONINSTRUCT, split="train")   # column 'text' = one block per row
blocks = [b for b in raw["text"] if b and b.strip()]
print("Blocks:", len(blocks))
print("Example:\\n", blocks[0][:300])"""),
    md("## 2. Group blocks into training chunks"),
    code("""chunks, cur = [], ""
for b in blocks:
    if len(cur) + len(b) < 800:
        cur += ("\\n\\n" + b) if cur else b
    else:
        if cur:
            chunks.append(cur)
        cur = b
if cur:
    chunks.append(cur)
print("Chunks:", len(chunks))"""),
    md("## 3. Load base model with Unsloth (4-bit QLoRA)"),
    code(MODEL_CFG),
    code("""from unsloth import FastLanguageModel
from datasets import Dataset
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = True,
)
EOS = tokenizer.eos_token
ds = Dataset.from_dict({"text": [c + EOS for c in chunks]})
print(ds)"""),
    md("## 4. Apply LoRA adapters"),
    code(LORA_STAGE1),
    md("## 5. Train on the raw text"),
    code("""from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,   # new TRL API (was 'tokenizer=')
    train_dataset = ds,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LEN,
        dataset_num_proc = 2,
        packing = True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10,      # more passes so the domain vocabulary actually sticks
        learning_rate = 5e-5,       # lower LR for continued pretraining
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_stage1",
        report_to = "none",
    ),
)
trainer_stats = trainer.train()"""),
    md("## 6. Push the Stage-1 adapter to the Hugging Face Hub"),
    code("""model.push_to_hub(ADAPTER_STAGE1, token=True)
tokenizer.push_to_hub(ADAPTER_STAGE1, token=True)
print("Pushed Stage-1 adapter to:", ADAPTER_STAGE1)"""),
    md("## 7. Quick sanity test (completion style, not Q&A yet)"),
    code("""FastLanguageModel.for_inference(model)
prompt = "The X_Product_Images table stores"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=60, temperature=0.3)
print(tokenizer.decode(out[0], skip_special_tokens=True))"""),
    md("### Next\nProceed to `instruction_finetuning.ipynb`."),
]

# ---------------------------------------------------------------------------
# Notebook 2: Instruction fine-tuning (SFT)
# ---------------------------------------------------------------------------
nb2 = [
    md("""# Stage 2 - Instruction Fine-Tuning (SFT)

**Goal:** teach the model to answer questions - map a natural-language question to either
the correct **table name** (schema discovery) or a valid **SQL query** (support queries).

Data: `ecomm-db-instruction` on the Hugging Face Hub (215 `{instruction, response}` examples)."""),
    code(INSTALL),
    code(GPU_CHECK),
    md("## 1. Config"),
    code(HF_CFG),
    code(HF_SAVE),
    md("## 2. Load model\nStart from the base model, or continue from the Stage-1 adapter for a domain-adapted start."),
    code(START_MODEL_CFG),
    code("""from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = START_MODEL,   # continue from the Stage-1 adapter (see START_MODEL above)
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = True,
)"""),
    code(LORA),
    md("## 3. Load and format the instruction dataset"),
    code(PROMPT_TEMPLATE),
    code("""from datasets import load_dataset
EOS = tokenizer.eos_token

def format_examples(batch):
    return {"text": [PROMPT.format(i, r) + EOS
                     for i, r in zip(batch["instruction"], batch["response"])]}

ds = load_dataset(DS_INSTRUCTION, split="train").map(format_examples, batched=True)
print(ds)
print(ds[0]["text"])"""),
    md("## 4. Train (SFT)"),
    code("""from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,   # new TRL API (was 'tokenizer=')
    train_dataset = ds,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LEN,
        dataset_num_proc = 2,
        packing = False,        # small dataset: keep one example per sequence
        padding_free = False,   # avoids the max_length/padding_free error when packing is off
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 10,  # small dataset: more passes needed for exact X_ schema recall
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_stage2",
        report_to = "none",
    ),
)
trainer_stats = trainer.train()"""),
    md("## 5. Push the SFT adapter to the Hugging Face Hub"),
    code("""model.push_to_hub(ADAPTER_STAGE2, token=True)
tokenizer.push_to_hub(ADAPTER_STAGE2, token=True)
print("Pushed SFT adapter to:", ADAPTER_STAGE2)"""),
    md("## 6. Inference after SFT"),
    code("""FastLanguageModel.for_inference(model)

def ask(question, max_new_tokens=128):
    text = PROMPT.format(question, "")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("### Answer:")[-1].strip()

for q in [
    "Which table stores product images?",
    "Give me a query to find unique orders for a customer.",
    "Find all shipments that have not been delivered.",
]:
    print("Q:", q); print("A:", ask(q)); print("-"*60)"""),
    code(SELF_CHECK),
    md("### Next\nProceed to `dpo_alignment.ipynb`."),
]

# ---------------------------------------------------------------------------
# Notebook 3: DPO alignment
# ---------------------------------------------------------------------------
nb3 = [
    md("""# Stage 3 - DPO Preference Alignment

**Goal:** push the SFT model to prefer *correct, specific* answers over *wrong / generic* ones
(hallucinated tables, missing DISTINCT, lazy SELECT *, wrong joins).

Data: `ecomm-db-preference` on the Hugging Face Hub (63 `{prompt, chosen, rejected}` examples)."""),
    code(INSTALL),
    code(GPU_CHECK),
    md("## 1. Config"),
    code(HF_CFG),
    code(HF_SAVE),
    md("## 2. Load the SFT model (Stage 2 adapter from the Hugging Face Hub)"),
    code(MODEL_CFG),
    code("""from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_STAGE2,   # SFT adapter pushed in Stage 2
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = True,
)"""),
    code("""from unsloth import FastLanguageModel
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)"""),
    md("## 3. Load and format the preference dataset"),
    code(PROMPT_TEMPLATE),
    code("""from datasets import load_dataset
EOS = tokenizer.eos_token

def format_pref(ex):
    return {
        "prompt": PROMPT.format(ex["prompt"], ""),
        "chosen": ex["chosen"] + EOS,
        "rejected": ex["rejected"] + EOS,
    }

ds = load_dataset(DS_PREFERENCE, split="train").map(format_pref)
print(ds); print(ds[0])"""),
    md("## 4. Configure and run DPO"),
    code("""from trl import DPOTrainer, DPOConfig
from unsloth import is_bfloat16_supported

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,          # LoRA -> no separate reference model needed (saves memory)
    processing_class = tokenizer,   # new TRL API (was 'tokenizer=')
    train_dataset = ds,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 2,
        learning_rate = 5e-5,
        beta = 0.1,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 3407,
        max_length = MAX_SEQ_LEN,
        max_prompt_length = 512,
        output_dir = "outputs_stage3",
        report_to = "none",
    ),
)
dpo_trainer.train()"""),
    md("## 5. Push the DPO-aligned adapter to the Hugging Face Hub"),
    code("""model.push_to_hub(ADAPTER_STAGE3, token=True)
tokenizer.push_to_hub(ADAPTER_STAGE3, token=True)
print("Pushed DPO adapter to:", ADAPTER_STAGE3)

# Optional: merge to a standalone 16-bit model and push it (no adapter needed to load)
# model.push_to_hub_merged(f"{HF_USER}/ecomm-db-final-merged", tokenizer, save_method='merged_16bit', token=True)"""),
    md("## 6. Test after DPO"),
    code("""FastLanguageModel.for_inference(model)

def ask(question, max_new_tokens=128):
    text = PROMPT.format(question, "")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("### Answer:")[-1].strip()

for q in [
    "Find the number of unique orders placed by customer 1001.",
    "Which table stores product images?",
    "Find the top 5 customers by total spend.",
]:
    print("Q:", q); print("A:", ask(q)); print("-"*60)"""),
    md("### Done\nThree adapters are now on the Hugging Face Hub (Stage 1 / 2 / 3). Use `src/inference.py` (point `--adapter` at the DPO repo) and fill in the `reports/` comparison tables."),
]

write("non_instruction_finetuning.ipynb", nb1)
write("instruction_finetuning.ipynb", nb2)
write("dpo_alignment.ipynb", nb3)
print("All notebooks generated.")
