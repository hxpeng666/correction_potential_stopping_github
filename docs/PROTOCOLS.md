# 实验协议与数据口径

## 共同原则

- LLM 冻结；只训练轻量 probe。
- split 单位是问题，不是 checkpoint；同一题的所有 checkpoints 只能属于一个 split。
- Dense rollout、checkpoint hidden、forced-answer 标签和最终评测按 `problem_id` 配对。
- Dense 使用采样：temperature 0.6、top-p 0.95、top-k 20、`do_sample=true`。
- forced answer 使用 greedy argmax，`do_sample=false`。
- 指标按 held-out 问题计算；校准不读取 held-out 标签。
- token cost 排除 prompt，包含实际保留的 reasoning token，以及真正停止时的一次 suffix/短答案。离线标签分支不计在线快速路径。

## 当前权威确定性协议

权威配置为 `configs/deepseek7b_deterministic_three_axis_ablation_v1.yaml`，
复现约束为 `docs/REPRODUCIBILITY_PROTOCOL_V1.md`。它在 DeepSeek 13K 数据口径
上进一步冻结以下选择：

- cap-hit Dense 答案：在 exact-13K prefix 后运行一次 greedy forced answer；
- 主特征：zero-based layer 16 的最后 checkpoint token hidden state与 6 个标量，
  合计 3590 维；
- 主优化：checkpoint-proper BCE；学习率 `5e-5`；normalized soft-min
  trajectory loss，`beta=0.5, lambda_tr=1`；
- 主校准：problem-level trajectory-envelope LTT，候选顺序来自 probe-train，
  独立 calibration 只负责认证与在安全集合中最大化 total generated-token
  reduction；不用经验预算 B，不用 wall time；
- 风险档位：`alpha = 0.005/0.01/0.02/0.03/0.05/0.10`，`delta=0.05`；
- MATH-500 与 AIME2024 是 OOD test，AIME 复用 MATH probe 和校准阈值。

正式 runner 仅接受 clean Git commit 与匹配的 runtime lock，并保存输入、标签、
特征、初始/最终权重及 score hash。旧的 seed 未锁、并发训练、经验 B 主表仅作
历史追溯，不得与当前权威表逐格混合比较。

## Qwen3-4B 历史实验

权威配置包括 `final_paper_*`、`gsm8k_full_checkpoint_schedule_ablation_v1.yaml` 和 `literature_methods_qwen3_4b_strict_v2.yaml`。

| 项目 | 值 |
|---|---|
| 模型 | Qwen/Qwen3-4B，冻结 |
| dtype / attention | float16 / SDPA（旧补充协议另保留 bf16 配置） |
| Dense budget | 4096 new tokens |
| forced-answer budget | 16 new tokens |
| checkpoint 主协议 | paragraph，全局允许、无范围过滤、无 checkpoint 则 Dense fallback |
| schedule 消融 | sentence、fixed budget、prefix stride、LYNX cue、paragraph、hybrid |
| hidden | zero-based layer 20，历史 layer 8/20/35 消融 |
| feature | `h + 6 scalars`，2566 维 |
| GSM8K | 1000 probe_train / 500 calibration / 1319 official test |
| MMLU-Pro | common-scope 1000 held-out；具体训练/校准来源由相应 config 锁定 |

Qwen 历史目录也保留早期 fixed/sentence 以及 wall-time replay 的代码。它们用于追溯旧表，不应与当前 paragraph/token-only 主口径混写。

## DeepSeek-R1-Distill-Qwen-7B 13K

权威配置：`configs/deepseek7b_main_v2.yaml`。

| 项目 | 值 |
|---|---|
| 模型 | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B，冻结 |
| dtype / attention | bfloat16 / SDPA |
| Dense budget | 13000 new tokens |
| forced-answer budget | 48 new tokens |
| checkpoint | paragraph，`\n\s*\n+`，无范围过滤，无点时 Dense fallback |
| hidden | zero-based layer 16，3584 维 |
| feature | `full_no_delta = h + 6 scalars`，3590 维 |
| probe architecture | 3590→384→96→1 |
| seed | 20260820 |

问题级数据划分：

| 数据集 | probe train | calibration | held-out / OOD test |
|---|---:|---:|---:|
| GSM8K | 1000 | 500 | 1319 |
| Hendrycks MATH train | 7 类×200 = 1400 | 7 类×100 = 700 | — |
| MATH-500 | — | — | 500 OOD |
| AIME 2024 | — | — | 30 OOD |

MATH 七类为 algebra、counting_and_probability、geometry、intermediate_algebra、number_theory、prealgebra、precalculus。MATH-500 与 AIME 共享同一套 MATH probe；AIME 必须复用在 MATH calibration 上冻结的权重和阈值，禁止 AIME 重训或重校准。

## DeepSeek selective-32K

权威配置：`configs/deepseek7b_main_v3_selective32k.yaml`。

- 目标 Dense budget 改为 32768；其他科学配置不变。
- 仅扩展旧 13K artifact 中 `reached_max_tokens=true` 的样本；未触顶样本逐文件等价迁移，生成内容保持不变。
- 扩展样本从 13K 缓存继续生成：重建 KV cache、精确 fast-forward 采样 RNG，再只生成 continuation。
- 旧 13K token、entropy、checkpoint hidden 和 forced answer 复用；新增 checkpoint 单独 replay checkpoint KV。
- 审计要求新轨迹前 13000 tokens 与旧 artifact 逐 token 一致。

测试集优先扩展协议另由 `TEST_ONLY_EXTENSION_MANIFEST.json` 在运行时生成；manifest 是数据产物，因此不随代码仓库发布。

## Fixed-budget frontier

对每题完整或冻结 Dense 长度 `T_i`，在：

$$
t_i(r)=\max(1,\lfloor rT_i\rfloor),\qquad r\in\{0.1,\ldots,0.9\}
$$

截断并 greedy forced-answer。所有题共享同一保留比例 `r`，training-free、无 calibration。报告实际 token reduction 时计入 suffix 和生成答案，因此 `r=0.5` 不必精确等于 50% reduction。13K 与 selective-32K frontier 有独立配置，不能混用 Dense reference。

## 结果字段

- `accuracy`：策略最终答案正确率；
- `dense_accuracy`：配对 Dense 基准正确率；
- `delta_accuracy = accuracy - dense_accuracy`；
- `token_reduction = 1 - mean(policy_tokens)/mean(dense_tokens)`；
- `lost_correct`：Dense 正确但早停结果错误的问题数；
- `helped`：Dense 错误但早停结果正确的问题数；
- `coverage`：至少一次提前停止的问题比例；
- `mean_checks`：每题实际执行的 hidden-state probe 数；
- `forced_answer_truncated`：短答案达到 forced-answer 上限仍未自然结束的计数。
