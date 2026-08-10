# 方法规范

## 状态与预测目标

对检查点 `t`，令 `c_t` 表示缓存中的强制作答是否正确，令 `f` 表示完整 Dense 轨迹的最终答案是否正确。四种状态分别为 `W→C`、`C→W`、`W→W` 和 `C→C`。主方法的标签为：

```text
y_t = 1[(not c_t) and f]
```

探针估计 `q_t=P(W→C|z_t)`。到达句子检查点时，如果 `q_t>tau`，则继续推理；否则停止推理，追加固定的最终答案后缀，并只生成一次答案。强制作答分支只用于离线标签构造，在线停止器不会查询这些分支。

## 特征

在第 20 层、隐藏宽度为 2,560 时：

```text
delta_h_t = h_t - h_previous
z_t = [h_t, delta_h_t, t, log(1+t), delta_t,
       entropy_tail8, ||delta_h_t||_2, cos(h_t, delta_h_t)]
```

第一个检查点的隐藏差分设为零。`entropy_tail8` 是最近八个 Dense 推理 token 的 top-20 下一 token 熵的平均值。`build_features` 和 `build_online_feature` 固定使用相同的特征顺序，并通过单元测试检查二者的一致性。

## 探针与损失函数

MLP 结构为 `5126→384→96→1`。第一层线性层后使用 LayerNorm，激活函数为 GELU，两处 dropout 概率分别为 0.15 和 0.10。

完整方法最小化检查点损失与轨迹损失之和。对每条含有 `W→C` 检查点的轨迹，轨迹损失使用 `beta=0.5` 的软最小值聚合这些检查点的 logit，并惩罚其中最低、最危险的 logit。`correction_bce` 消融仅移除该轨迹损失，其余设置保持不变。

## 受控预测目标

- 正确性：检查点强制作答是否正确。
- 一致性：检查点强制作答是否等于 Dense 最终答案；两个缺失答案不视为一致。
- 最后切换：检查点是否严格位于最后一次答案变化之后，其中也包括最后一个检查点答案到 Dense 答案之间的变化。
- 修正潜力：检查点答案错误且 Dense 最终答案正确。

正确性、一致性和最后切换使用检查点 BCE，并在高分时停止；修正潜力在低分时停止。前三者是统一实现框架中的受控预测目标，不代表对其他系统的论文级原样复现。

## 风险校准

阈值网格由 101 个 calibration 分数分位点和一个完全不提前停止的 sentinel 组成。每个阈值都以轨迹为单位回放，并选择第一个满足停止条件的检查点。

当前论文主协议使用 calibration lost-correct 的历史经验绝对预算 `B={0,1,2,4,10}`。对每个 B，在 `W→C count<=B` 的阈值中选择 calibration replay latency 最低者；并依次以 token reduction 和 coverage 作为平局判据。Strict、Balanced、Aggressive 分别是 B=1、2、4。B 是有限 calibration 集上的经验事件数，不是总体风险的置信上界。Bonferroni/Clopper–Pearson 代码如保留，只能作为 `formal-certified-*` 补充结果，不能替代或混写当前主协议。

另行在同一阈值曲线上选择 calibration coverage 最接近 30%、40%、50%、60%、70%、80%、90% 的工作点。held-out coverage 无需等于 calibration 目标，held-out 指标从不参与阈值或 epoch 选择。

## 缺少合法检查点

如果 Dense 轨迹短于最小长度，或者不存在合法句子边界，则该问题回退到 Dense。它仍然保留在准确率、coverage、风险、延迟和 bootstrap 统计的分母中。
