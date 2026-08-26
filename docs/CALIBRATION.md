# 阈值校准策略

## 基本回放

在独立 calibration 问题上，对约 101 个分数分位点构造候选阈值。每个候选必须按“首次越过阈值即停止”的完整问题轨迹回放，不能把 checkpoints 当作独立样本统计。

校准约束的基本事件是问题级 lost-correct：Dense 最终正确，但所选早停答案错误。`B` 是整个 calibration 问题集合上允许的 lost-correct 问题数，不是每道题各允许 B 个错误，也不是 checkpoint 数量预算。

## 仓库保留的经验策略

1. `empirical_B`：仅要求 `N_LC(tau) <= B`，在可行阈值中最大化 calibration token reduction。
2. `empirical_B + accuracy floor`：除 B 外还要求 calibration accuracy 不低于 Dense accuracy 1 个百分点；这是 `train_deepseek7b_ablation_v1.py` 的历史主表口径。
3. fixed shared threshold：预声明同一数值阈值，不从 calibration 选择；可不用 calibration，但分数跨方法/seed 不同尺度，通常不适合作为主比较。
4. coverage-matched：让各方法达到相近停止覆盖率，适合诊断目标本身，不提供风险保证。

因此，“固定 B 消融”是否带 1pp floor 必须在表头说明：历史五组主训练脚本带 floor；`run_deepseek7b_calibration_strategy_token_v2.py` 中重新评估的 pure empirical-B 不带 floor。

## Problem-level formal calibrators

`run_deepseek7b_calibration_strategy_token_v2.py` 实现：

- `bonferroni_cp`：对所有候选阈值做 Bonferroni 修正的 Clopper–Pearson 上界；
- `fixed_sequence_ltt`：按从保守到激进的预声明顺序做 Learn-Then-Test，首次失败即停止扩张；
- `trajectory_first_failure_conformal`：校准每题首次危险停止边界的 conformal rank；
- `trajectory_envelope_ltt`：对整条危险轨迹的 first-failure envelope 做 problem-level LTT；
- `lynx_class_conditional`：LYNX 风格 class-conditional split conformal 对照。

对每个风险水平 `alpha` 和置信失败概率 `delta`：

1. 只用 calibration 问题判断阈值可行性；
2. 在可行候选中最大化 calibration token reduction；
3. 若 token reduction 相同，优先更高 coverage，再选更保守的稳定次序；
4. MATH-500 与 AIME 只用于最终报告，不选择 family、alpha 或 threshold。

当前推荐流程是先在预声明的 problem-level PAC family 间，仅凭 BCE+trajectory 的 calibration token-reduction 与 bootstrap 选择稳定性选 family；主风险点使用 `alpha=0.01`，balanced 点使用 `alpha=0.03`。推荐逻辑写入 `RECOMMENDATION.json`，不硬编码某次实验胜者。

## OOD 限制

MATH calibration 到 MATH-500/AIME 是 zero-shot threshold transfer。即使在 MATH calibration 上有 exchangeability 下的有限样本风险保证，该保证也不会自动跨分布成立。OOD 表应同时报告 lost-correct、helped、accuracy delta、token reduction 与 Dense fallback，而不能只报净 accuracy。

## 为什么用 token reduction

当前研究不把 wall time 作为阈值选择目标。所有新校准策略使用：

$$
1-\frac{\operatorname{mean}(\text{policy reasoning + actual final-answer tokens})}
{\operatorname{mean}(\text{Dense reasoning tokens})}.
$$

旧 Qwen 代码中的 A100 replay wall-time 字段仅为历史可追溯实现，不得改写成当前 token-only 实测。
