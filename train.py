"""最小 DPO：Base + SFT adapter + DPO LoRA，用 TRL DPOTrainer。

Preference Data -> Policy/Reference logprob -> DPO Loss -> 更新 Policy
DPO 不需要 Reward Model、不需要 Critic、不需要 rollout、不需要 GAE。

两个模型角色：
  Policy    = Base + SFT adapter（merge）+ 新的 DPO LoRA   <- 训练
  Reference = ref_model=None，TRL 用 use_adapter() 把 DPO LoRA 关掉现算  <- 冻结

运行：python train.py
"""

import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
BASE = "Qwen/Qwen3-0.6B"
SFT_ADAPTER = HERE.parent / "sft_tool_call" / "output"
DATA = HERE / "train.jsonl"
OUTPUT = HERE / "output"


def show_sample(tok, row):
    """一条 preference 数据过完 chat template 之后长什么样。"""
    print("=" * 78)
    print("[原始 preference 数据]  conversational 格式，由 chat template 渲染")
    print("  prompt   :", row["prompt"])
    print("  chosen   :", row["chosen"])
    print("  rejected :", row["rejected"])

    for name, resp in (("chosen", row["chosen"]), ("rejected", row["rejected"])):
        prompt_text = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(row["prompt"] + resp, tokenize=False)
        n_prompt = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
        n_full = len(tok(full_text, add_special_tokens=False)["input_ids"])
        print(f"\n  [{name}] 总长 {n_full} token = prompt {n_prompt} + response {n_full - n_prompt}")
        print(f"         {full_text!r}")
    print("\nlogπ(response|prompt) 就是把 response 那一段的逐 token logprob 加起来。")


def main():
    # ---- 1. 加载 preference dataset ----
    ds = load_dataset("json", data_files=str(DATA), split="train")
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    show_sample(tok, json.loads(DATA.read_text(encoding="utf-8").splitlines()[0]))

    # ---- 2. Policy = Base + SFT adapter（merge），DPO 阶段再加 LoRA ----
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16), str(SFT_ADAPTER)
    ).merge_and_unload()
    print("=" * 78)
    print("[两个模型]")
    print("  Policy    : Base + SFT adapter(merged) + 新的 DPO LoRA   <- 训练")
    print("  Reference : ref_model=None，TRL 关掉 DPO LoRA 现算 SFT 模型  <- 冻结")
    print("  （没有 Reward Model，没有 Critic）")

    # ---- 3. DPO LoRA（和 SFT / PPO 完全一致的配置）----
    dpo_lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )

    # ---- 4. 训练配置 ----
    args = DPOConfig(
        output_dir=str(OUTPUT),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        beta=0.1,
        loss_type="sigmoid",
        max_length=512,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    # ---- 5. DPOTrainer ----
    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=dpo_lora,
    )

    p = trainer.model
    trainable = sum(x.numel() for x in p.parameters() if x.requires_grad)
    print(f"[配置] beta={args.beta} loss_type={args.loss_type} "
          f"batch_size={args.per_device_train_batch_size} x grad_accum="
          f"{args.gradient_accumulation_steps} -> 有效 batch size = "
          f"{args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"[DPO LoRA] 可训练参数 = {trainable:,}")

    trainer.train()
    trainer.save_model(str(OUTPUT))
    print(f"\nDPO adapter 已保存到 {OUTPUT}")


if __name__ == "__main__":
    main()
