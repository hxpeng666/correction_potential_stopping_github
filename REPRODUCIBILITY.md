# 可复现性检查清单

## 输入与协议冻结

- [ ] 选择的 YAML 与 `configs/PROTOCOL_REGISTRY.json` 中 protocol ID 一致。
- [ ] 模型 revision、dtype、attention backend、hidden layer 与 config 一致，基础模型冻结。
- [ ] Dense 与 forced-answer decoding 分开记录；不要把 Dense 的 temperature 误用于 greedy forced answer。
- [ ] split 单位为问题；同一题的 checkpoints 不跨 probe train、calibration、held-out。
- [ ] Qwen 与 DeepSeek 的 4K/16、13K/48、selective-32K/48 协议不混写。
- [ ] 数据准备 manifest 包含样本 ID、来源 revision、顺序与 SHA256。

## 采集

- [ ] 各方法共享同一条 Dense trajectory；schedule 消融只改变 checkpoint 集合。
- [ ] reasoning 终点、paragraph regex、无 checkpoint Dense fallback 与 config 一致。
- [ ] forced-answer suffix、greedy 策略与 token 上限写入每个 artifact。
- [ ] hidden shape 与 row 数严格对齐，不含 NaN/Inf。
- [ ] current/Dense prediction 用数据集相应 verifier 重新计算后与缓存字段一致。
- [ ] `--resume` 只跳过 fingerprint 一致且 `status=complete` 的 artifact。
- [ ] selective-32K 扩展样本前 13000 tokens 与 13K 来源逐 token 相同；未触顶迁移样本内容不变。

## Probe

- [ ] StandardScaler 只拟合 probe-train internal-fit 问题。
- [ ] internal model-selection 也按问题切分。
- [ ] 五个目标共用 feature、architecture、optimizer、split、schedule 和缓存。
- [ ] BCE+trajectory 使用 normalized soft-min、beta 0.5、lambda 1；probe manifest 明确记录。
- [ ] `tests/test_normalized_softmin_v1.py` 通过。

## 校准与测试

- [ ] threshold 只由 calibration 问题选择。
- [ ] 每个候选按首次停止的完整轨迹规则回放。
- [ ] 表头声明 pure B 是否另带 calibration accuracy floor；不得混淆。
- [ ] token-objective 策略不读取 wall time。
- [ ] formal calibrator 使用 problem-level 事件和预声明 alpha/delta/family。
- [ ] MATH-500/AIME 不选择 probe、calibrator family、alpha 或 threshold。
- [ ] AIME 复用 MATH probe 与 threshold，不重训、不重校准。

## 报告

- [ ] held-out sample ID 对齐。
- [ ] accuracy、Dense accuracy、delta accuracy、token reduction、lost-correct、helped、coverage 分开报告。
- [ ] token cost 排除 prompt、包含真实停止时的一次 suffix/short answer。
- [ ] offline label branches 不计在线成本；在线答案探测式方法的额外 probe tokens 必须计入。
- [ ] OOD 表明确说明 exchangeability 风险保证不自动迁移。
- [ ] fixed-budget frontier 的比例是每题相对 Dense 长度，不是全数据集统一绝对 token 数。
