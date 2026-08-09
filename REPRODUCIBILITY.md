# 可复现性检查清单

只有完成以下检查，结果才可以作为论文实验结果使用。

## 不可变输入

- [ ] Qwen3-4B revision 与配置一致，且 `inspect_qwen3` 检查通过。
- [ ] 使用 FP16 和 SDPA，不进行量化，也不训练基础模型。
- [ ] 全局随机种子严格为 `20260803`。
- [ ] 数据集 revision 与 `prepare_final_paper_data.py` 中固定的哈希一致。
- [ ] 已先生成父划分，再运行 `prepare_final_paper_single_seed_scope.py`，scope 指纹与仓库清单一致。
- [ ] GSM8K 为 2,000/1,000/1,319，其中 held-out 是完整 official test。
- [ ] MMLU 为 2,000/1,000/1,000；MMLU-1k 覆盖全部 57 个学科，每科 17 或 18 题。
- [ ] MMLU 每个学科恰好使用 5 个 dev demonstrations；held-out 选择不读取模型输出或标签。

## 公共缓存

- [ ] 每个样本只生成一次 Dense 轨迹，并由所有方法共享。
- [ ] Direct 和每个句子及固定检查点的强制作答分支均存在。
- [ ] 断点恢复只接受完全一致的协议指纹。
- [ ] 每个生成随机种子都等于固定五元组计算出的随机种子。
- [ ] Dense token 列表长度等于 `reasoning_tokens`。
- [ ] hidden 张量不含 NaN/Inf，形状为 `[检查点数, 1, 2560]`，且与记录逐行对齐。
- [ ] 检查点键没有重复，并且恰好等于句子与固定检查点集合的并集。
- [ ] `CACHE_AUDIT.json` 的状态为 `PASS`，没有缺失或额外 sample ID。

## 训练与策略校准

- [ ] StandardScaler 只在 probe-train 的 fit problems 上拟合。
- [ ] 早停只使用固定的 probe-train 内部 validation 划分。
- [ ] 所有 adaptive 方法共享特征、架构、优化器、数据划分、检查点计划和公共缓存。
- [ ] 正确性、一致性、最后切换和修正潜力 BCE 使用各自声明的损失。
- [ ] 主方法使用修正潜力标签和 `beta=0.5` 的轨迹软最小值保护。
- [ ] 阈值只使用 1,000 个 policy calibration problems。
- [ ] 正式风险上界包含有限阈值网格的多重比较校正。
- [ ] 风险约束不可行时报告 Dense fallback，不放宽阈值。

## 时间与报告

- [ ] 在声明的目标 GPU 上预热后，以单请求方式采集 timing calibration。
- [ ] 时间无效或被污染的样本从时间模型中排除，但不改变其语义有效性。
- [ ] 时间模型 fit/validation 不包含 held-out 任务样本。
- [ ] Dense 和检查点成本的 validation MAPE 不超过 5%，p95 相对误差不超过 10%。
- [ ] Direct 或强制作答 worker 的时间不进入 replay 成本。
- [ ] 每个已检查的检查点都按声明方式计入停止器开销。
- [ ] 每个延迟列均标记为“目标设备单请求回放估计延迟”。
- [ ] 每种方法的 held-out sample ID 完全对齐：GSM8K 为完整 official test，MMLU 为冻结的 MMLU-1k。
- [ ] 配对 bootstrap 重复 10,000 次，所有方法使用相同的重采样 ID。
- [ ] 净准确率与 lost-correct 风险分开报告。
- [ ] 最终结论遵守预先声明的正面或负面判定标准，不使用 held-out 调参。
- [ ] 报告将 MMLU 结果明确标作 `MMLU-1k`，不声称为完整 14,042 题 MMLU test。
