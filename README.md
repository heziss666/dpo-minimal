# dpo-minimal

最小可运行的 **DPO** 实验，学习链的第四个也是最后一个环节：

```
Preference Data -> Policy/Reference logprob -> DPO Loss -> 更新 Policy
```

对比 PPO-RLHF，DPO **不需要**：

```
× Reward Model
× Critic
× rollout
× GAE
× temperature / response_length / gamma / lam / cliprange / vf_coef
```

**只需要两个模型角色**：

| 模型 | 来源 | 是否训练 |
|---|---|---|
| Policy | Base + SFT adapter（merge）再加 DPO LoRA | 训练 |
| Reference | `ref_model=None`，TRL 用 `use_adapter()` 关掉 DPO LoRA 现算 | 冻结 |

Reference 没有单独占显存：DPO LoRA 加在 merge 后的 SFT 模型上，关掉它就是 π_ref。

## 数据

**直接复用 `reward-model-minimal` 的数据，一行没改。** 因为 RM 和 DPO 需要的都是 $(x, y^+, y^-)$：

```json
{"prompt":   [{"role":"system",...},{"role":"user","content":"北京今天天气怎么样？"}],
 "chosen":   [{"role":"assistant","content":"{\"tool\":\"weather\",\"arguments\":{\"city\":\"北京\"}}"}],
 "rejected": [{"role":"assistant","content":"{\"tool\":\"calculator\",\"arguments\":{\"expression\":\"北京天气\"}}"}]}
```

于是 RM 和 DPO 可以用**完全相同的 preference data**，横向比较非常干净。

## 安装与运行

只需要 transformers + peft + **trl ≥ 0.24**（`DPOTrainer` 是 TRL 的稳定 API，不像 `PPOTrainer` 已被移除）。本项目用前面 SFT / RM 项目的主环境即可，**不需要 PPO 那个 venv**。

```powershell
python evaluate.py --base                # DPO 之前的 SFT 策略（基线）
python train.py                          # DPO 训练，约 85 秒
python evaluate.py --adapter ./output    # DPO 之后
```

版本（跑通时实测）：`trl 1.14.0` / `transformers 5.14.1` / `peft 0.19.1` / `torch 2.13.0.dev+cu132`。

## 结果

| 指标 | SFT（DPO 前） | DPO 后 |
|---|---|---|
| JSON Valid | 90.0% | 90.0% |
| Tool Accuracy | 70.0% | **75.0%** |
| Arguments Accuracy | 45.0% | **50.0%** |
| Exact Match | 45.0% | **50.0%** |
| Preference Accuracy | 90.0% | **95.0%** |
| Preference Margin | 15.56 | **17.45** |

六项**全部没有下降**，其中五项上升。

固定样例（`帮我查一下杭州明天天气`）：

| | logπ(chosen) | logπ(rejected) | margin |
|---|---|---|---|
| SFT | -0.65 | -9.46 | +8.81 |
| DPO | **-0.54** | **-11.00** | **+10.46** |

**chosen 的概率上升、rejected 的概率下降**——这正是 DPO loss 要的效果。训练日志里也能看到同一个趋势：

```
rewards/accuracies (训练集): 0.65 -> 1.00
rewards/margins            : 0.018 -> 0.216
logps/chosen   -3.34 -> -2.73
logps/rejected -20.24 -> -18.38
```

30 个 optimizer step（80/8 × 3 epoch），84 秒。

## 三个方法的横向比较

相同的 SFT 起点、相同的 preference 数据、相同的 LoRA 配置（`q_proj`/`v_proj`，r=16）：

| 方法 | Reward Model | Critic | Rollout | Preference Data | Exact Match |
|---|---:|---:|---:|---:|---:|
| SFT | ✗ | ✗ | ✗ | ✗ | 45.0% |
| PPO-RLHF | ✓ | ✓ | ✓ | RM 阶段用 | 30.0% |
| **DPO** | ✗ | ✗ | ✗ | ✓ 直接用 | **50.0%** |

| 指标 | SFT | PPO-RLHF | DPO |
|---|---|---|---|
| JSON Valid | 90.0% | 95.0% | 90.0% |
| Tool Accuracy | 70.0% | 55.0% | **75.0%** |
| Arguments Accuracy | 45.0% | 40.0% | **50.0%** |
| Exact Match | 45.0% | 30.0% | **50.0%** |

PPO 那一列取两次实验中较好的那次（`--vf-coef 1.0`）。

**结论很明显：在这个只有 80 条 preference pair 的小规模场景里，DPO 用三分之一的组件数打赢了 PPO-RLHF。**

原因不神秘：PPO 需要在线 rollout，再靠 Critic 估 advantage；Critic 学不动时 advantage 就是噪声，策略会被推坏（详见 `ppo-rlhf-minimal` 的诊断）。DPO 完全绕开了这条链，直接用数据里现成的 chosen/rejected 做离线优化，没有 Critic、没有采样方差、也没有那些难调的超参。

**但要诚实**：这不代表 DPO 全面优于 PPO。DPO 是 offline 的，只在给定偏好数据分布内优化；PPO 能探索数据之外的回答，理论上上限更高——前提是 reward/critic 足够可靠。小数据小模型下，稳定性压倒一切。

## 核心代码位置

| 步骤 | 位置 |
|---|---|
| 一条 preference 数据的 chat template 渲染 | `train.py:32` |
| Policy = Base + SFT adapter（merge） | `train.py:60` |
| DPO LoRA 配置 | `train.py:68` |
| DPOConfig（`beta=0.1` / `loss_type="sigmoid"`） | `train.py:74` |
| DPOTrainer：`model` / `ref_model=None` / `peft_config` | `train.py:90` |
| forward -> DPO loss -> backward -> optimizer.step() | `train.py:107` |
| logπ(y\|x)：逐 token logprob 求和 | `evaluate.py:93` |
| 生成式任务指标 | `evaluate.py:146` |
| Preference Accuracy / Margin | `evaluate.py:151` |

DPO loss 本身在 TRL 内部，没有自己实现：

```python
# trl/trainer/dpo_trainer.py:1425-1430, 1458-1463
chosen_logratios   = chosen_logps   - ref_chosen_logps     # logπθ(y+|x) - logπref(y+|x)
rejected_logratios = rejected_logps - ref_rejected_logps
chosen_scores      = chosen_logratios                      # loss_type="sigmoid" 时就是 logratio 本身
rejected_scores    = rejected_logratios
delta_score        = chosen_scores - rejected_scores
per_sequence_loss  = -F.logsigmoid(self.beta * delta_score)
```

也就是 $L=-\log\sigma\bigl(\beta\,[m_\theta-m_{ref}]\bigr)$。
