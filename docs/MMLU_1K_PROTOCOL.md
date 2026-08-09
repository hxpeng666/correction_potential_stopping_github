# MMLU-1k 降级评测协议依据

## 决策

本轮实验使用单随机种子 `20260803`。GSM8K 仍评测完整官方 test 1,319 题；MMLU 使用从官方 test 中预先冻结的 1,000 题分层子集，并明确记为 **MMLU-1k**，不得写成完整 MMLU test。

MMLU-1k 覆盖全部 57 个学科：规范学科顺序中的前 31 科各取 18 题，其余 26 科各取 17 题，总计 1,000 题。每个学科内部使用 `sha256(20260803:mmlu:heldout:problem_id)` 排序后取固定前缀。选择过程不读取模型输出、正确性、推理长度、checkpoint 或延迟。

Probe training 为 GSM8K 1,000 题和 MMLU 1,000 题；threshold calibration 各 500 题。held-out 不参与 probe 训练、特征标准化和阈值选择。MMLU probe/calibration 来自按 57 学科路由后的 `auxiliary_train`，而 held-out 来自 official test，因此当前结果必须明确标记为 distribution-shift，不能声称经验风险已在同分布 calibration 上得到验证。

## 同类工作依据

1. Dong、Qin 与 Shah 的 learned-stopping 对照研究 *When Does Learning to Stop Help?* 在主表中对 MMLU-Pro 使用 `N=800`，并同时使用 GSM8K `N=1000`。这说明昂贵的逐 checkpoint stopping 研究通常采用固定中等规模评测集，而非必然运行完整大测试集。链接：https://arxiv.org/abs/2606.30852
2. *Demystifying Long Chain-of-Thought Reasoning* 为效率采用 1,000 条 MMLU-Pro test 的 i.i.d. 子集，并尽量保持原始类别分布。链接：https://openreview.net/pdf/1b6de9c6455dbdf78c235f2540914e872ebfede2.pdf
3. *FROST: Factual Reasoning via Optimized Stochastic Trajectories* 在经典 57-subject MMLU 上报告 1,000 条分层评测样本。链接：https://openreview.net/pdf/b8fe0570e2fe872d6dddc9096e567a151107a916.pdf

## 报告限制

- MMLU-1k 的 overall micro accuracy 是均衡子集上的 micro accuracy，不等价于按完整 MMLU 原始题量加权的 micro accuracy。
- 57-subject macro accuracy、四大类别结果和每学科结果仍可报告，但单学科仅有 17–18 题，置信区间会较宽。
- 所有方法必须使用完全相同的 1,000 个 sample ID；bootstrap 在学科内分层并对全部方法复用相同重采样索引。
- 最终结论必须同时展示 GSM8K 完整 test 与 MMLU-1k；不得声称已经完成 MMLU 14,042 题完整评测。
