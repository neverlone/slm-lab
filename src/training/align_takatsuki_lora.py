"""
Takatsuki-3B and Takatsuki-8B Conversational LoRA Alignment Engine.
Fine-tunes base architectures with PEFT/LoRA on curated multi-turn uncensored dialogue.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def train_lora_alignment(
    model_name_or_path: str = "Qwen/Qwen2.5-3B",
    output_dir: str = "models/takatsuki_3b_lora",
    merged_dir: str = "models/takatsuki_3b_merged",
    dataset_name: str = "teknium/OpenHermes-2.5",
    max_samples: int = 50000,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    max_seq_len: int = 2048,
):
    if not PEFT_AVAILABLE:
        print("Installing peft and trl...")
        os.system("uv pip install peft trl")

    print(f"Loading Base Architecture: {model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.bfloat16,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading alignment dataset: {dataset_name} ({max_samples:,} samples)...")
    raw_dataset = load_dataset(dataset_name, split=f"train[:{max_samples}]")

    def format_and_tokenize(batch):
        texts = []
        for conv in batch["conversations"]:
            formatted = []
            for turn in conv:
                role = "user" if turn.get("from") == "human" else "assistant"
                content = turn.get("value", "")
                formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            texts.append("\n".join(formatted))
        
        return tokenizer(
            texts,
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
        )

    tokenized_dataset = raw_dataset.map(format_and_tokenize, batched=True, remove_columns=raw_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("Starting Takatsuki LoRA Fine-Tuning Run...")
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")

    # Merge LoRA back into base weights for GGUF export
    print(f"Merging LoRA weights into standalone model: {merged_dir}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    merged_model = PeftModel.from_pretrained(base_model, output_dir).merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Exported merged model to {merged_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B", help="Base model for alignment")
    parser.add_argument("--out_dir", type=str, default="models/takatsuki_3b_merged", help="Output directory")
    args = parser.parse_args()
    train_lora_alignment(model_name_or_path=args.model, merged_dir=args.out_dir)
