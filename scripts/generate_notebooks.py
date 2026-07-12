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

# Merged 16-bit models that chain the stages: Stage N merges its LoRA into the
# base, and Stage N+1 loads that merged model and adds a FRESH, fully-trainable
# LoRA on top (bootcamp Class-22's "merge then add new adapter" approach).
MERGED_STAGE1 = f"{HF_USER}/ecomm-db-stage1-merged"
MERGED_STAGE2 = f"{HF_USER}/ecomm-db-stage2-merged"
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

# If the loaded model ALREADY has a LoRA adapter (i.e. we continued from a
# previous stage's adapter), keep training THAT adapter instead of adding a new,
# conflicting one. Otherwise (a fresh base model) attach a new LoRA adapter.
if getattr(model, "peft_config", None):
    print("Model already has a LoRA adapter - continuing to train it.")
    try:
        FastLanguageModel.for_training(model)
    except Exception:
        pass
    # A loaded adapter is often frozen (inference mode); re-enable grads so
    # training actually updates it.
    for _n, _p in model.named_parameters():
        if "lora_" in _n:
            _p.requires_grad_(True)
else:
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
    )

# Sanity: there MUST be trainable parameters, else trainer.train() does nothing
# and the model stays generic. This catches a frozen/misloaded adapter early.
model.print_trainable_parameters()
_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert _trainable > 0, "No trainable parameters - adapter is frozen; training would do nothing."
print("Trainable params:", _trainable)"""

# Stage 2 config. Follows the teaching: non-instruction (Stage 1) -> instruction
# (Stage 2). We start from the MERGED Stage-1 model (domain adaptation baked into
# the weights) and add a FRESH, fully-trainable LoRA for instruction tuning.
START_MODEL_CFG = MODEL_CFG + """

# Stage 2 starting point (chained pipeline):
#   START_MODEL = MERGED_STAGE1  -> continue from Stage-1 domain adaptation (the taught flow)
#   START_MODEL = MODEL_NAME     -> skip Stage 1 and train on the plain base model
START_MODEL = MERGED_STAGE1"""

# Merged-reload evaluation (HR-project pattern) -----------------------------
# Evaluating the in-memory LoRA adapter directly can silently fall back to
# base-model output in some Unsloth/transformers versions (training happens, loss
# drops to ~0, but generation ignores the adapter). Baking the LoRA into the
# weights with save_pretrained_merged and RELOADING that merged model removes the
# ambiguity - the training is now part of the actual weights, so generation cannot
# bypass it. This is the approach that reliably works in the HR project.
def merged_eval(stage_tag, questions):
    q_lines = "\n".join(f"    {q!r}," for q in questions)
    return f'''# Bake the trained LoRA into the weights, then RELOAD the merged model and
# evaluate THAT (not the in-memory adapter). See generator notes: this is the
# HR-project pattern that reliably reflects the training at generation time.
model.save_pretrained_merged("{stage_tag}_merged_local", tokenizer, save_method="merged_16bit")

eval_model, eval_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "{stage_tag}_merged_local",
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = False,   # full 16-bit reload = exact recall of memorized answers
)
FastLanguageModel.for_inference(eval_model)

def ask(question, max_new_tokens=128):
    text = PROMPT.format(question, "")
    inputs = eval_tokenizer(text, return_tensors="pt").to("cuda")
    out = eval_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return eval_tokenizer.decode(out[0], skip_special_tokens=True).split("### Answer:")[-1].strip()

for q in [
{q_lines}
]:
    print("Q:", q); print("A:", ask(q)); print("-"*60)'''


# Stage 1 completion-style sanity check on the reloaded merged model.
EVAL_STAGE1 = """# Bake the trained LoRA into the weights, then RELOAD the merged model and test IT.
# Evaluating the in-memory adapter directly can silently show base-model behavior in
# some Unsloth/transformers versions; merging removes that ambiguity (HR pattern).
model.save_pretrained_merged("stage1_merged_local", tokenizer, save_method="merged_16bit")

eval_model, eval_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "stage1_merged_local",
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = False,
)
FastLanguageModel.for_inference(eval_model)
prompt = "The X_Product_Images table stores"
inputs = eval_tokenizer(prompt, return_tensors="pt").to("cuda")
out = eval_model.generate(**inputs, max_new_tokens=60, do_sample=False)
print(eval_tokenizer.decode(out[0], skip_special_tokens=True))"""

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

print("\\nMODEL LEARNED THE SCHEMA:", all_ok)
assert all_ok, (
    "Verbatim training examples were NOT reproduced by the reloaded merged model. "
    "This means training did not actually update the weights for these mappings. "
    "Check: (1) you ran the training cell in THIS session, (2) the dataset loaded "
    "correctly (ds[0] shows the X_ answers), (3) epochs/LR are high enough, and "
    "re-run training, then this cell, without restarting the runtime in between."
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
    code(LORA),
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
    md("## 6. Quick sanity test (completion style, not Q&A yet)\n"
       "We save the trained model MERGED and reload it, then test the reloaded model. "
       "This guarantees we evaluate the trained weights (not an in-memory adapter that "
       "can silently fall back to base-model output)."),
    code(EVAL_STAGE1),
    md("## 7. Push the Stage-1 adapter + merged model to the Hugging Face Hub"),
    code("""model.push_to_hub(ADAPTER_STAGE1, token=True)
tokenizer.push_to_hub(ADAPTER_STAGE1, token=True)
print("Pushed Stage-1 adapter to:", ADAPTER_STAGE1)

# Also push a MERGED 16-bit model so Stage 2 can load it as its base and add a
# FRESH, fully-trainable LoRA on top (chained: non-instruction -> instruction).
model.push_to_hub_merged(MERGED_STAGE1, tokenizer, save_method="merged_16bit", token=True)
print("Pushed merged Stage-1 model to:", MERGED_STAGE1)"""),
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
    md("## 2. Load model\nContinue from the merged Stage-1 model (domain-adapted), then add a fresh LoRA. Set `START_MODEL = MODEL_NAME` to skip Stage 1."),
    code(START_MODEL_CFG),
    code("""from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = START_MODEL,   # merged Stage-1 model (see START_MODEL above)
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
    md("## 5. Inference + self-check on the reloaded MERGED model\n"
       "We save the trained LoRA merged into the weights, reload that merged model, and "
       "evaluate it. This is the HR-project pattern: it guarantees generation reflects "
       "the training (evaluating the in-memory adapter can silently show base output)."),
    code(merged_eval("stage2", [
        "Which table stores product images?",
        "Give me a query to find unique orders for a customer.",
        "Find all shipments that have not been delivered.",
    ])),
    code(SELF_CHECK),
    md("## 6. Push the SFT adapter + merged model to the Hugging Face Hub\n"
       "Run this only after the self-check above prints `ADAPTER ACTIVE + LEARNED: True`."),
    code("""model.push_to_hub(ADAPTER_STAGE2, token=True)
tokenizer.push_to_hub(ADAPTER_STAGE2, token=True)
print("Pushed SFT adapter to:", ADAPTER_STAGE2)

# Also push a MERGED 16-bit model so Stage 3 (DPO) can load it as its base and
# add a fresh LoRA on top.
model.push_to_hub_merged(MERGED_STAGE2, tokenizer, save_method="merged_16bit", token=True)
print("Pushed merged Stage-2 model to:", MERGED_STAGE2)"""),
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
    md("## 2. Load the merged SFT model (Stage 2) from the Hugging Face Hub"),
    code(MODEL_CFG),
    code("""from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MERGED_STAGE2,   # merged SFT model from Stage 2 (fresh LoRA added below)
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = True,
)"""),
    code(LORA),
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
    md("## 5. Test after DPO on the reloaded MERGED model\n"
       "Save the trained model merged, reload it, and evaluate the reloaded model so "
       "generation always reflects the training (HR-project pattern)."),
    code(merged_eval("stage3", [
        "Find the number of unique orders placed by customer 1001.",
        "Which table stores product images?",
        "Find the top 5 customers by total spend.",
    ])),
    md("## 6. Push the DPO-aligned adapter to the Hugging Face Hub"),
    code("""model.push_to_hub(ADAPTER_STAGE3, token=True)
tokenizer.push_to_hub(ADAPTER_STAGE3, token=True)
print("Pushed DPO adapter to:", ADAPTER_STAGE3)

# Optional: merge to a standalone 16-bit model and push it (no adapter needed to load)
# model.push_to_hub_merged(f"{HF_USER}/ecomm-db-final-merged", tokenizer, save_method='merged_16bit', token=True)"""),
    md("### Done\nThree adapters are now on the Hugging Face Hub (Stage 1 / 2 / 3). Use `src/inference.py` (point `--adapter` at the DPO repo) and fill in the `reports/` comparison tables."),
]

write("non_instruction_finetuning.ipynb", nb1)
write("instruction_finetuning.ipynb", nb2)
write("dpo_alignment.ipynb", nb3)
print("All notebooks generated.")
