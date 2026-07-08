"""Simple inference script for the final DPO-aligned e-commerce DB assistant.

Usage:
    python src/inference.py --adapter /path/to/stage3_dpo_adapter
    python src/inference.py --adapter <path> --question "Which table stores product images?"

Run this on a machine with a CUDA GPU (e.g. Colab). It loads the saved LoRA
adapter on top of the base model and answers questions - either the table name
(schema discovery) or a SQL query (support queries).
"""
import argparse

PROMPT = (
    "Below is a question about the client e-commerce database schema. "
    "Write a response that correctly answers it, giving the exact table name(s) or a valid SQL query.\n\n"
    "### Question:\n{}\n\n### Answer:\n{}"
)


def load_model(adapter_path: str, max_seq_len: int = 2048):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,   # a saved adapter dir also carries the base model ref
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    text = PROMPT.format(question, "")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=False,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Answer:")[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="E-commerce DB assistant inference")
    parser.add_argument("--adapter", required=True, help="Path to the saved DPO (or SFT) adapter directory")
    parser.add_argument("--question", default=None, help="Ask a single question and exit")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)

    if args.question:
        print(generate_answer(model, tokenizer, args.question, args.max_new_tokens))
        return

    print("E-commerce DB Assistant (type 'exit' to quit)")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        print("Answer:", generate_answer(model, tokenizer, question, args.max_new_tokens))


if __name__ == "__main__":
    main()
