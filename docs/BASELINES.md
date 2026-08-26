# 对照目标、checkpoint 与相关工作复现

## 五个受控目标

所有目标共用同一 Dense trajectory、问题划分、checkpoint schedule、feature、MLP architecture 和 calibration 候选网格；只改变监督定义或 loss：

| 名称 | 监督目标 | 停止分数方向 |
|---|---|---|
| correctness | `P(c_t=1 | z_t)` | 高分更适合停止 |
| consistency | `P(a_t=a_T | z_t)` | 高分更适合停止 |
| last_switch | 当前是否越过最后一次答案变化 | 高分更适合停止 |
| bce | `P(c_t=0 and f=1 | z_t)` | 低风险才停止 |
| bce_traj | 同 bce + normalized trajectory loss | 低风险才停止 |

这五组是目标层面的公平消融，不等同于论文原生 LTS/LYNX 复现。

## Checkpoint schedules

- `paragraph`：空行边界；当前主协议，无范围过滤。
- `sentence`：换行或句末标点，minimum gap 防止重复密集触发。
- `fixed_budget`：预声明 token 位置。
- `prefix_stride`：固定 token stride。
- `lynx_cue`：在 `hmm`、`wait`、`alternatively` 等 cue 前检查。
- `hybrid`：优先语义边界，但用最大等待 token 数补点。
- `native`：相关论文自己的 checkpoint 定义。

paragraph 通常是 sentence 的较稀疏语义子集，但“候选点更多”不保证更优：首次越阈值策略会累积多次误停机会，固定风险校准也可能被迫选择更保守阈值。

## Learn-to-Stop (LTS)

严格复现代码在 `collect_literature_method_data_v1.py` 与 `train_evaluate_literature_method_v1.py`：

- 原生 checkpoint：NLTK sentence end；另提供 paragraph common-checkpoint 版本。
- 表示：checkpoint 处最终归一化 last-layer hidden。
- 标签：从当前 chunk 开始，后续 forced answers 是否始终等于 terminal forced answer（stable suffix / last switch），而不是 W→C correction potential。
- probe：单层 LSTM，hidden size 128，Adam，learning rate 5e-4；验证集按 F1 选模型。

## LYNX

- 原生 checkpoint：`hmm`、`wait`、`alternatively` cue 之前；另提供 paragraph 版本。
- 原生风格 feature：多个指定层的 event hidden 拼接。
- 每个 event 离线 forced exit，标签是 stop-now answer correctness。
- probe：256→64 两隐藏层 ReLU MLP、标准化、class-balanced loss。
- class-conditional split conformal，在线使用 singleton-positive predictive set 决策。
- 无 cue 的 train/calibration 题按严格配置在 thinking span 70% 放 synthetic checkpoint；held-out 无 cue 题 Dense fallback，避免利用测试答案补点。

## Self-verification

- 按 reasoning-path cue 分 chunk，取 chunk endpoint hidden；
- 两层加权 MLP 预测 intermediate answer correctness；
- 原文 Gemini 2.0 Flash 的抽取/判定在 common-scope 复现中由同一冻结 Qwen3-4B 替代，最终 held-out correctness 仍由冻结数据集 parser 判断；
- 因 labeler 被替代，这属于方法层面对齐、模型/标签器口径统一的复现，不应宣称原论文 exact system reproduction。

## Thought Calibration

- 原生 checkpoint 为含 `wait` 或 `but` 的双换行 reasoning section；
- 对 step tokens 的最终归一化 last-layer hidden 做均值池化，再 PCA 到 256 维并训练线性 probe；
- supervised 与 consistent 两种 target；概率使用长度 10 的 smoothing window；
- Learn-Then-Test fixed-sequence binomial-tail calibration。
- common-scope 版本使用已有 4B 数据，不调用原文独立 Qwen3-32B labeler，因此同样明确标注为标签器替代复现。

## 公平比较边界

每个相关工作有两种 schedule：论文原生 checkpoint 和 paragraph common checkpoint。前者检验原方法，后者控制“checkpoint 选得好不好”。不要给论文原生方法附加本方法的 B 概念；B 只用于另行报告的 common empirical-risk operating points。答案探测生成的 token 成本应在需要在线探测时计入；纯 hidden-state readout 的离线 forced-answer 标签不计在线快速路径。
