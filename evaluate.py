"""评测 DPO：SFT 策略（DPO 前）vs SFT+DPO。

生成式任务指标（沿用前三个项目）：
  JSON Valid / Tool Accuracy / Arguments Accuracy / Exact Match
DPO 特有指标（直接比概率，不需要生成）：
  Preference Accuracy       = logπ(chosen) > logπ(rejected) 的比例
  Average Preference Margin = mean(logπ(chosen) - logπ(rejected))

用法：
  python evaluate.py --base                # DPO 之前的 SFT 策略（基线）
  python evaluate.py --adapter ./output    # DPO 之后
  python evaluate.py                       # 不传参数时自动检测 ./output
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
BASE = "Qwen/Qwen3-0.6B"
SFT_ADAPTER = HERE.parent / "sft_tool_call" / "output"
DATA = HERE / "test.jsonl"
DEFAULT_ADAPTER = HERE / "output"

CHAT_KWARGS = {"enable_thinking": False}
MAX_NEW_TOKENS = 64
# 固定样例：DPO 前后各看一次，观察 logprob 与 preference margin 的变化
DEMO = (
    [{"role": "system", "content": "你是一个工具调用助手。根据用户需求选择工具并提取参数，只输出合法JSON，不要解释。"},
     {"role": "user", "content": "帮我查一下杭州明天天气"}],
    [{"role": "assistant", "content": '{"tool":"weather","arguments":{"city":"杭州"}}'}],
    [{"role": "assistant", "content": '{"tool":"search","arguments":{"query":"杭州天气"}}'}],
)


def extract_json(text):
    """从模型输出里抽出第一个完整的 {...} 对象；失败返回 None。"""
    text = re.sub(r"<think>.*?</think>", "", text.replace("<|im_end|>", ""), flags=re.S)
    for m in re.finditer(r"\{", text):
        depth, in_str, esc = 0, False, False
        for i in range(m.start(), len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[m.start():i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def load_policy(dpo_adapter):
    """Base + SFT adapter（merge） [+ DPO adapter]。"""
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(base, str(SFT_ADAPTER)).merge_and_unload()
    if dpo_adapter:
        policy = PeftModel.from_pretrained(policy, str(dpo_adapter))
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    return tok, policy.to("cuda").eval()


def generate(tok, policy, prompt_msgs):
    """确定性生成（do_sample=False）。"""
    text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(policy.device)
    with torch.no_grad():
        out = policy.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def response_logprob(tok, policy, prompt_msgs, response_msgs):
    """logπ(y|x) = Σ_t logπ(a_t | x, a_<t)，只累加 response 部分的 token。"""
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
    full_text = tok.apply_chat_template(prompt_msgs + response_msgs, tokenize=False)
    prompt_len = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
    ids = tok(full_text, add_special_tokens=False)["input_ids"]

    input_ids = torch.tensor([ids], device=policy.device)
    with torch.no_grad():
        logits = policy(input_ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    # 位置 t 的 token 由位置 t-1 的 logits 预测，所以只在 response 区间取 t-1
    pos = torch.arange(prompt_len, len(ids), device=policy.device)
    targets = torch.tensor(ids, device=policy.device)[pos]
    return logprobs[pos - 1, targets].sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="只测 DPO 之前的 SFT 策略")
    ap.add_argument("--adapter", default=None, help="DPO LoRA adapter 目录")
    a = ap.parse_args()

    adapter = None if a.base else (a.adapter or (str(DEFAULT_ADAPTER) if DEFAULT_ADAPTER.exists() else None))
    label = f"SFT + DPO({adapter})" if adapter else "SFT（DPO 之前）"
    print(f"===== 评测对象：{label} =====")

    tok, policy = load_policy(adapter)
    rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]

    stats = {"json_valid": 0, "tool": 0, "args": 0, "exact": 0}
    pref_ok, margins, failures = 0, [], []
    for row in rows:
        target = json.loads(row["chosen"][0]["content"])
        # --- 生成式任务指标 ---
        raw = generate(tok, policy, row["prompt"])
        pred = extract_json(raw)
        if pred is not None:
            stats["json_valid"] += 1
            stats["tool"] += pred.get("tool") == target["tool"]
            stats["args"] += pred.get("arguments") == target["arguments"]
            stats["exact"] += pred == target
        if pred != target:
            failures.append((row["prompt"], target, raw, pred))
        # --- DPO 特有的概率指标 ---
        lp_c = response_logprob(tok, policy, row["prompt"], row["chosen"])
        lp_r = response_logprob(tok, policy, row["prompt"], row["rejected"])
        pref_ok += lp_c > lp_r
        margins.append(lp_c - lp_r)

    n = len(rows)
    print(f"\nTotal: {n}")
    print("--- 生成式任务指标 ---")
    print(f"JSON Valid Rate    : {stats['json_valid'] / n:6.1%}")
    print(f"Tool Accuracy      : {stats['tool'] / n:6.1%}")
    print(f"Arguments Accuracy : {stats['args'] / n:6.1%}")
    print(f"Exact Match        : {stats['exact'] / n:6.1%}")
    print("--- DPO 特有的概率指标（不生成，直接比 logπ）---")
    print(f"Preference Accuracy: {pref_ok / n:6.1%}   ({pref_ok}/{n})")
    print(f"Preference Margin  : {sum(margins) / n:6.2f}   (mean logπ(chosen) - logπ(rejected))")

    print("\n===== 固定样例 =====")
    prompt, chosen, rejected = DEMO
    print(f"Prompt   : {prompt[1]['content']}")
    print(f"  generation : {generate(tok, policy, prompt)!r}")
    lp_c = response_logprob(tok, policy, prompt, chosen)
    lp_r = response_logprob(tok, policy, prompt, rejected)
    print(f"  logπ(chosen)   = {lp_c:8.2f}   {chosen[0]['content']}")
    print(f"  logπ(rejected) = {lp_r:8.2f}   {rejected[0]['content']}")
    print(f"  preference margin = {lp_c - lp_r:+.2f}   {'✓ 偏向 chosen' if lp_c > lp_r else '✗ 偏向 rejected'}")

    if failures:
        print(f"\n===== 未完全匹配的样例（共 {len(failures)} 条，展示前 8 条）=====")
        for prompt_msgs, tgt, raw, pred in failures[:8]:
            print(f"\nInput      : {prompt_msgs[1]['content']}")
            print(f"Target     : {json.dumps(tgt, ensure_ascii=False)}")
            print(f"Prediction : {json.dumps(pred, ensure_ascii=False) if pred else raw!r}")


if __name__ == "__main__":
    main()
