# 方法：Correction-Potential Stopping

## 1. 离线监督

冻结 LLM 先生成完整 Dense reasoning trajectory。对 checkpoint `t` 的已有 reasoning prefix 追加固定后缀：

```text
\n</think>\n\n\boxed{
```

forced-answer branch 使用 greedy argmax；Qwen 历史协议最多 16 tokens，DeepSeek 当前协议最多 48 tokens。它只回答“如果现在停止，模型会输出什么答案”，不参与在线检查。

设标准答案、checkpoint 答案与 Dense 最终答案分别为 `a*`、`a_t`、`a_T`：

$$
c_t=\mathbb 1[a_t=a^*],\qquad f=\mathbb 1[a_T=a^*].
$$

修正潜力标签：

$$
y_t=\mathbb 1[c_t=0\land f=1].
$$

只有 W→C 是危险停止事件：

| checkpoint → Dense | 标签 | 提前停止含义 |
|---|---:|---|
| W→C | 1 | 破坏未来本可发生的正确修正 |
| C→W | 0 | 可能挽救后续损坏 |
| W→W | 0 | 继续推理未带来 Dense 正确性收益 |
| C→C | 0 | 当前已正确 |

## 2. Checkpoint 与特征

主部署 schedule 为 paragraph：使用正则 `\n\s*\n+` 在线检测空行边界，不施加 token 范围过滤；没有 paragraph checkpoint 的题回退 Dense。固定 token、sentence、prefix stride、LYNX cue 和 hybrid 是受控消融。

从指定 zero-based layer 读取 checkpoint 最后一个 reasoning token 的 hidden state：

$$
h_t\in\mathbb R^d.
$$

对前一 checkpoint `t^-`：

$$
\Delta h_t=h_t-h_{t^-},\qquad \Delta h_t=0\ \text{for the first checkpoint}.
$$

完整 `Delta h` 不拼接到当前主 MLP，仅计算两个标量：

$$
n_t=\lVert\Delta h_t\rVert_2,
$$

$$
s_t=\frac{h_t^\top\Delta h_t}
{\lVert h_t\rVert_2\lVert\Delta h_t\rVert_2+\epsilon}.
$$

再加入 `t`、`log(1+t)`、`Delta t=t-t^-`，以及正常 reasoning decode 中 top-20 logits 重新归一化后、最近 8 tokens 的平均熵 `H_t`。最终：

$$
z_t=[h_t;t;\log(1+t);\Delta t;H_t;n_t;s_t].
$$

- Qwen3-4B：`d=2560`，主 layer 20，`dim(z)=2566`；
- DeepSeek-R1-Distill-Qwen-7B：`d=3584`，主 layer 16，`dim(z)=3590`。

标准化统计只在 probe-train 的 internal-fit 问题上拟合，同一问题的 checkpoints 不会跨 internal fit/model-selection 两侧。

## 3. Probe

两个模型使用同形状的轻量 readout（输入宽度随 LLM hidden width 改变）：

```text
Linear(input, 384)
→ LayerNorm → GELU → Dropout(0.15)
→ Linear(384, 96) → GELU → Dropout(0.10)
→ Linear(96, 1)
```

输出：

$$
q_t=\sigma(a_t)=P(c_t=0\land f=1\mid z_t).
$$

训练使用 AdamW，learning rate `2e-4`，weight decay `1e-3`，trajectory batch 为 24 个完整问题，gradient clipping `2.0`，最多 24 epochs，patience 6。

## 4. BCE 与 normalized trajectory loss

检查点级加权 BCE：

$$
L_{pt}=\frac1M\sum_t w_t\,\operatorname{BCEWithLogits}(a_t,y_t).
$$

危险 W→C checkpoint 权重为 1.5；其他 checkpoint 根据剩余 Dense 比例加权，越早且安全的点得到更高节省价值权重。

对第 `i` 条轨迹的危险集合：

$$
D_i=\{t:y_{i,t}=1\}.
$$

当前修复后的最弱点聚合使用 normalized log-mean-exp：

$$
\widetilde m_i=-\beta\left[
\log\sum_{t\in D_i}\exp(-a_{i,t}/\beta)-\log|D_i|
\right].
$$

$$
L_{tr}=\frac1{|\mathcal I_+|}\sum_{i\in\mathcal I_+}
\operatorname{softplus}(-\widetilde m_i).
$$

$$
L=L_{pt}+\lambda_{tr}L_{tr},\qquad
\beta=0.5,\quad\lambda_{tr}=1.
$$

减去 `log|D_i|` 后，当一条危险轨迹上所有 logits 相同时，trajectory loss 不再随危险 checkpoint 数增长。回归测试位于 `tests/test_normalized_softmin_v1.py`。

## 5. 在线策略

`q_t` 是“停止会破坏未来修正”的风险分数，因此：

$$
\pi_t=\begin{cases}
\mathrm{continue},&q_t>\tau,\\
\mathrm{stop},&q_t\le\tau.
\end{cases}
$$

取首个满足停止条件的 checkpoint；若无满足点则回退 Dense。停止后只追加一次最终答案 suffix 并生成一次短答案。在线阶段没有逐 checkpoint forced-answer、多样本投票或答案解析，所以 stopper 是 pure hidden-state one-step readout。

阈值只用独立 calibration 问题选择，held-out/OOD 测试不参与模型、阈值或校准策略选择。校准策略见 `docs/CALIBRATION.md`。
