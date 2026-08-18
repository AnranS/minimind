# MiniMind-3 Notebooks

这组交互笔记本按当前主线（`minimind-3`）拆解仓库代码，方便对照 `README.md` 和 `trainer/` 里的正式脚本学习。

请在 `notebook/` 目录下启动 Jupyter / VS Code，确保工作目录就是本目录，这样相对路径（`../model`、`./toydata`）才能对上。

## 阅读顺序

| 笔记本 | 对应主线 | 说明 |
|---|---|---|
| `0_tokenizer.ipynb` | `trainer/train_tokenizer.py`、`model/tokenizer.json` | BPE + ByteLevel；官方词表已自带，本本只做演示 |
| `1_model.ipynb` | `model/model_minimind.py` | Config、RMSNorm、RoPE/YaRN、GQA、QK-Norm、SwiGLU、MoE（无 shared expert） |
| `2_dataset.ipynb` | `dataset/lm_dataset.py` | Pretrain / SFT / DPO / RLAIF / Agent 数据格式与 Dataset |
| `3_pretrain.ipynb` | `trainer/train_pretrain.py` | `model(input_ids, labels=...)`，`loss + aux_loss` |
| `4_sft.ipynb` | `trainer/train_full_sft.py` | 对话模板、Tool Call、思考标签已混入主线 SFT |
| `5_dpo.ipynb` | `trainer/train_dpo.py` | RLHF 偏好优化 |
| `6_lora.ipynb` | `trainer/train_lora.py`、`model/model_lora.py` | 低秩增量；可用 `scripts/convert_model.py` 合并回基模 |
| `7_distill.ipynb` | `trainer/train_distillation.py` | 白盒蒸馏：`α·CE + (1-α)·T²·KL` |
| `8_thinking.ipynb` | chat_template + `open_thinking` | 替换已删除的独立 Reason 训练 |
| `9_rlaif.ipynb` | `trainer/train_grpo.py` | GRPO / CISPO 的 group-relative advantage |
| `10_agent.ipynb` | `trainer/train_agent.py` | 多轮 Tool-Use + mock tool + `gt` |

## 和正式训练的差别

这些笔记本用 `toydata/` 里的几条样本、把模型缩到 `hidden_size=64, num_hidden_layers=1`，只演示流程，**不能替代** `cd trainer && python train_xxx.py`。

完整复现请看仓库根目录 `README.md`：数据用 `pretrain_t2t(_mini).jsonl` / `sft_t2t(_mini).jsonl` / `rlaif.jsonl` / `agent_rl.jsonl`，脚本都在 `trainer/`。

## 环境

与主仓库相同：`pip install -r requirements.txt`。另外需要能跑 Jupyter 的环境（VS Code、JupyterLab 均可）。
