# MiniMind-3 学习计划（约 14 天）

在自己的 fork 上按这个顺序学即可，不必给上游提 PR。

默认每天 1.5–3 小时。笔记本只负责看懂流程；真正出能聊天的模型，仍然要 `cd trainer && python train_xxx.py`。

请在 `notebook/` 目录打开 Jupyter / VS Code，保证工作目录就是本目录，这样 `../model`、`./toydata` 才能对上。

每天收工问自己三件事：

1. 这一步输入是什么，输出权重叫什么？
2. loss 打在哪些 token 上？
3. 如果删掉这一步，模型会缺什么能力？

---

## 先记住这条主线

```text
预训练 Pretrain          监督微调 SFT              后面都是选修
text 接龙            →   学会对话模板         →   LoRA / DPO / 蒸馏
只学语言和知识            顺带学会 tool / think     RLAIF / Agent
pretrain_*.pth           full_sft_*.pth
```

- **预训练**数据是 `pretrain_t2t(_mini).jsonl`，格式只有 `{"text": "..."}`，没有角色，也没有 tool / think。
- **SFT 必须做**，不能省。数据是 `sft_t2t(_mini).jsonl`。
- 不再单独做的，是 SFT **之后**再开一轮 Tool Call、再开一轮 Reason。

### 为什么 SFT 之后一般不用再单独训 Tool Call / Reason

这句话说的不是「不用做 SFT」，而是「SFT 这一轮已经把这两件事一起做了」。

旧主线大概是：

```text
Pretrain → SFT 对话 → 再训一轮 Tool Call → 再训一轮 Reason（train_reason.py，产出 reason_*.pth）
```

MiniMind-3 改成：

```text
Pretrain → SFT（对话 + Tool Call + 思考样本混在同一份 sft_t2t 里）→ 可选 RLAIF / Agent
```

原因分两边：

**Tool Call 进了 SFT 数据。**  
`sft_t2t` / `sft_t2t_mini` 里已经混了约 10 万条从 `qwen3-4b` 采的工具调用样本。格式是 OpenAI 风格的多轮消息：`system.tools`、`assistant.tool_calls`、`role=tool`。训练时 `chat_template` 会把它们展开成 `<tool_call>...</tool_call>` / `<tool_response>...</tool_response>`。所以跑完 `train_full_sft.py` 得到的 `full_sft_*.pth`，本身就会基本的 tool call，不必再为工具单独开一轮 SFT。

**Reason 不再单独产一个权重。**  
仓库从 2026-03 起删了 `train_reason.py` 和 `reason_*.pth`。思考不再靠「再训一个只会 `<think>…</think><answer>…</answer>` 的模型」，而是：

- 模板层用 `<think>` 预留位置；
- 推理时用 `open_thinking=0/1` 决定是直接答，还是先写出思考；
- 训练时在 SFT / RLAIF 里混合空 think、带 `reasoning_content` 的样本，以及 `thinking_ratio` 采样。

同一套 `full_sft` 权重既能直答也能想，只是开关不同。

**不要记成：** tool / think 已经进了预训练，所以不用 SFT。  
**应该记成：** 预训练只学接龙；SFT 学对话，并且把 tool / think 的格式一起灌进去。后面的 RLAIF / Agent 是用奖励强化「该不该调用、答案对不对」，不是补上 SFT 没教过的格式。

局限：tool 目前大约 10 个模拟工具，谈不上泛化；tool 和显式思考同时开时还不稳，因为缺联合样本。

---

## 第 1 天：建立地图（0.5–1h）

先读仓库根目录 `README.md`：

1. 项目介绍 + 2026-04-01 更新
2. Tokenizer / 数据介绍
3. 模型结构表（64M Dense / 198M-A64M MoE）
4. 「主要训练」：Pretrain → SFT 是必须的

读完应能回答：

- 词表为什么只有 6400：小模型里 embedding + `lm_head` 参数占比很大，大词表会把模型撑肥，中间层反而变瘦。6400 是保体积，不是中文切得更好。
- 为什么 SFT 之后一般不用再单独训 Tool Call / Reason：见上一节，它们混在 **SFT 数据**里，不是预训练里。
- 训练脚本都在 `trainer/`，不在仓库根目录。

---

## 第 2 天：Tokenizer（`0_tokenizer.ipynb`）

**目标：** 搞清 BPE + ChatML，不要自己训一套官方词表。

- 先加载 `../model`，看 `bos/eos/pad`、`<think>`、`<tool_call>`
- 再跑玩具 BPE，体会 `train_from_iterator` → `tokenizer.json`
- 对照 `trainer/train_tokenizer.py` 和 `model/tokenizer_config.json`

**过关：** 能解释 `<|im_start|>user ... <|im_end|>` 怎么拼出来；知道为什么不要重训官方词表。

---

## 第 3–4 天：模型（`1_model.ipynb` + `model/model_minimind.py`）

按模块对着源码看：

| 顺序 | 模块 | 盯住什么 |
|------|------|----------|
| 1 | `MiniMindConfig` | `hidden_size` / `num_hidden_layers`，不是旧的 `dim` / `n_layers` |
| 2 | RMSNorm | Pre-Norm，没有减均值 |
| 3 | RoPE + YaRN | 实数 cos/sin；YaRN 只在推理外推时长上下文 |
| 4 | Attention | GQA（8Q/4KV）、QK-Norm、Flash Attention |
| 5 | SwiGLU FFN | `silu(gate) * up` |
| 6 | `MiniMindForCausalLM` | `forward(..., labels=)` 内部算 CE，外加 `aux_loss` |
| 7 | MoE | 4 experts / top-1，没有 shared expert |

**过关：** 合上文件能画出一层 Block；能说出 embedding 和 `lm_head` 为什么要 tie。

---

## 第 5 天：数据（`2_dataset.ipynb`）

1. Pretrain：`{text}` → `(input_ids, labels)`，pad 的 label 是 `-100`
2. SFT：`conversations`，loss 只打 assistant；已含 tools / think
3. DPO：`chosen` / `rejected`
4. RLAIF：只返回 prompt，留给 rollout
5. Agent：`messages + tools + gt`

**过关：** 打开一条 SFT tool 样本，能指出来模板里 `<tool_call>` / `<tool_response>` 是哪来的。

---

## 第 6–8 天：主线训练（必须）

### 第 6 天：`3_pretrain.ipynb`

搞懂 `loss = res.loss + res.aux_loss`，再读 `trainer/train_pretrain.py`。

### 第 7 天：`4_sft.ipynb`

和预训练同一套 `model(..., labels=)`，换的是 Dataset 和 mask。tool / think 已经在这份数据里。

### 第 8 天：有 GPU 再真训 Zero

数据放到 `../dataset/`：

- 最快：`pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`
- 完整：`pretrain_t2t.jsonl` + `sft_t2t.jsonl`

```bash
cd trainer
python train_pretrain.py
python train_full_sft.py
```

然后回仓库根目录：

```bash
python eval_llm.py --weight pretrain
python eval_llm.py --weight full_sft
python scripts/eval_toolcall.py --weight full_sft
```

没 GPU 就停在 notebook + 读脚本。

**过关：** 能独立讲清 Pretrain 和 SFT 各解决什么问题；知道权重落在 `out/`。

---

## 第 9–10 天：选修微调（`5/6/7`）

三选一深挖，另外两本浏览即可。

- `6_lora.ipynb`：只训低秩分支，适合垂域。合并看 `scripts/convert_model.py`
- `5_dpo.ipynb`：chosen vs rejected，ref 冻结。更常改善礼貌，不是智力
- `7_distill.ipynb`：黑盒 = 继续 SFT；白盒 = `α·CE + (1-α)·T²·KL`

**过关：** 能说清什么时候用 LoRA 而不是 full SFT。

---

## 第 11–13 天：思考、RL、Agent

### 第 11 天：`8_thinking.ipynb`

`open_thinking=0/1` 模板差在哪；为什么独立 Reason 训练被删了。

```bash
python eval_llm.py --load_from ./minimind-3 --open_thinking 1
```

### 第 12 天：`9_rlaif.ipynb`

组相对 advantage；clip + KL。浏览 `trainer/train_grpo.py`、`trainer/rollout_engine.py`，先不追求自己复现完整 rollout。

### 第 13 天：`10_agent.ipynb`

解析 `<tool_call>` → mock 执行 → 回填结果；`gt` 怎么当可校验奖励。对照 `trainer/train_agent.py`。

**过关：** 能画一张图：SFT 教会格式，GRPO / Agent 用奖励强化「该不该调用、答案对不对」。

---

## 第 14 天：推理与对外接口（可选）

- `eval_llm.py`：本地对话、`open_thinking`
- `scripts/serve_openai_api.py`、`scripts/chat_api.py`
- `scripts/web_demo.py`

---

## 两周节奏

```text
D1      读 README 地图
D2      Tokenizer
D3–D4   模型（最重要）
D5      五种数据
D6–D8   Pretrain + SFT（有卡就真训 mini）
D9–D10  LoRA / DPO / Distill 选深一门
D11–D13 Thinking → RLAIF → Agent
D14     推理 / API
```

有卡：D8 是唯一必须上正式脚本的节点。  
没卡：全程 notebook + 读 `trainer/` 也够入门。

---

## 别踩的坑

1. 不要重训官方 tokenizer，否则后面权重全废。
2. 不要把 notebook 里 `hidden_size=64` 的结果当成模型效果。
3. 不要再找 `train_reason.py` / `<answer>` 标签，那是旧主线。
4. 对照代码时认新名字：`MiniMindForCausalLM`、`MiniMindConfig`、`hidden_size`。
5. 正式训练一律 `cd trainer`，数据在 `../dataset/`。
6. tool / think 在 **SFT 数据**里，不在预训练数据里。
