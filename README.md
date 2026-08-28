# Correction-Potential Stopping

冻结推理模型的轻量 early-stopping 实验代码。仓库同时保留两条可区分、可审计的实验线：

- 历史 Qwen3-4B：GSM8K / MMLU-Pro，checkpoint schedule、layer、特征、标签、trajectory loss、LTS/LYNX 等基线复现；
- 当前 DeepSeek-R1-Distill-Qwen-7B：GSM8K、Hendrycks MATH 训练/校准，MATH-500 与 AIME 2024 OOD 测试，13K 主协议、选择性 32K 扩展和 fixed-budget frontier。

仓库不包含模型权重、数据缓存、hidden-state 缓存、probe 权重、结果表、运行日志或服务器 GPU 调度器。科学配置采用仓库相对路径；单机、多卡或集群调度由使用者在外层实现。

## 核心方法

在 checkpoint `t` 离线强制生成短答案，定义当前答案正确性 `c_t` 和 Dense 最终正确性 `f`。主目标为：

```text
y_t = 1[c_t = 0 and f = 1]
q_t = P(y_t = 1 | z_t)
```

`q_t` 高表示此时停止可能截断未来本可发生的 W→C 修正，因此继续；`q_t` 低于校准阈值时停止。forced-answer 只用于离线标签，不进入在线检查路径；部署时只读取已计算的 hidden state、六个标量动态特征和一个小型 MLP。

五个受控目标均已提供：

1. `correctness`：预测当前 forced answer 是否正确；
2. `consistency`：预测当前答案是否等于 Dense 最终答案；
3. `last_switch`：预测当前是否越过最后一次答案变化；
4. `bce`：预测 W→C 修正潜力；
5. `bce_traj`：W→C BCE 加归一化 trajectory 最弱点保护。

方法公式与在线/离线边界见 [docs/METHOD.md](docs/METHOD.md)。

## 冻结协议一览

| 配置 | Qwen3-4B 历史协议 | DeepSeek-7B 13K | DeepSeek-7B selective-32K |
|---|---:|---:|---:|
| Dense decoding | sample, T=0.6, top-p=0.95, top-k=20 | 相同 | 相同，13K 前缀保持一致后续跑 |
| Dense max new tokens | 4096 | 13000 | 32768 |
| Forced answer | greedy | greedy | greedy |
| Forced answer max tokens | 16 | 48 | 48 |
| 主 checkpoint | paragraph，无范围过滤 | paragraph，无范围过滤 | paragraph，无范围过滤 |
| 无 checkpoint | Dense fallback | Dense fallback | Dense fallback |
| hidden layer | zero-based 20 | zero-based 16 | zero-based 16 |
| hidden width / feature width | 2560 / 2566 | 3584 / 3590 | 3584 / 3590 |
| trajectory | normalized soft-min, beta=0.5, lambda=1 | 相同 | 相同 |

全部数据划分和口径见 [docs/PROTOCOLS.md](docs/PROTOCOLS.md)，机器可读入口为 [configs/PROTOCOL_REGISTRY.json](configs/PROTOCOL_REGISTRY.json)。

## 安装

建议 Linux、Python 3.10+、CUDA GPU。验证环境使用 PyTorch 2.6 与 Transformers 4.53.3。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 先按机器 CUDA 版本安装 torch，再安装其余依赖
pip install -r requirements.txt
```

模型默认路径：

```text
models/Qwen3-4B/
models/DeepSeek-R1-Distill-Qwen-7B/
```

也可修改相应 YAML 的 `model.local_path`。代码不会上传或捆绑权重。

## DeepSeek-7B 复现入口

准备问题级划分：

```bash
python scripts/prepare_deepseek7b_data_v1.py \
  --project-root . \
  --math-snapshot /path/to/hendrycks_math \
  --math500-snapshot /path/to/math500 \
  --legacy-math-split-root /path/to/legacy_math_split \
  --aime-parquet /path/to/aime2024.parquet \
  --output-root data/deepseek7b_main_v2
```

正式采集 paragraph checkpoint（单卡示例，可用互斥 shard 并行）：

```bash
python scripts/run_committed_experiment_v1.py \
  --name deepseek7b-deterministic-collection-worker0 \
  --output-root results/deepseek7b_deterministic_recollection_v1/launch_worker0 \
  --config configs/deepseek7b_deterministic_recollection_v1.yaml -- \
  python scripts/collect_deepseek7b_paragraph_v1.py \
    --config configs/deepseek7b_deterministic_recollection_v1.yaml \
    --gpu 0 --worker-id local-0 --shard-index 0 --num-shards 1 --resume
```

正式采集默认要求已提交的 runtime lock；旧的未锁配置只能显式加
`--allow-unlocked-legacy` 进行历史检查，其输出不能进入新结果表。Dense
仍按论文协议使用采样，但每道题的 RNG seed 由全局 seed 与 problem ID
稳定派生，因此与 worker 数、shard 顺序无关。

训练五组 probe 时，对 `correctness/consistency/last_switch` 使用 `--loss bce`；对 `bce` 使用 `--method correction --loss bce`；对 `bce_traj` 使用：

```bash
python scripts/train_deepseek7b_ablation_v1.py \
  --config configs/deepseek7b_deterministic_grader_pair_v2.yaml \
  --raw-root results/deepseek7b_main_v2/cache \
  --heldout-root results/deepseek7b_main_v2/cache/math500/heldout \
  --output outputs/deepseek7b/bce_traj \
  --method correction --loss bce_traj \
  --trajectory-aggregation normalized_softmin \
  --trajectory-beta 0.5 --trajectory-weight 1 \
  --schedule paragraph --actual-schedule-label paragraph \
  --layer 16 --feature-kind full_no_delta --seed 0 --gpu 0
```

MATH-500 训练出的 probe/阈值在 AIME 2024 上只复用、不重训、不重校准：

```bash
python scripts/evaluate_deepseek7b_ood_v2.py \
  --dataset aime --source-probe outputs/deepseek7b/bce_traj \
  --heldout-root results/deepseek7b_main_v2/cache/aime/heldout \
  --runtime-lock configs/runtime_a100_torch271_cuda126_v1.json \
  --output outputs/deepseek7b/aime_bce_traj --gpu 0
```

选择性 32K 延展、13K/32K fixed relative-budget frontier 分别由对应 `configs/deepseek7b_*` 配置和同名 collect/migrate/summarize 脚本执行。增量延展会验证旧 13K token 前缀身份，不把未触顶样本重新生成。

## Qwen 与相关工作入口

- Qwen checkpoint schedule：`collect_full_checkpoint_schedule_v1.py` + `gsm8k_full_checkpoint_schedule_ablation_v1.yaml`；
- greedy forced-answer 重采：`collect_greedy_forced_answer_v1.py`；
- normalized trajectory：`train_controlled_label_normalized_trajectory_v1.py`；
- layer/feature/paragraph 表示消融：`materialize_*`、`train_paragraph_representation_probe_v1.py`；
- LTS、LYNX、self-verification、Thought Calibration：`collect_literature_method_data_v1.py` 与 `train_evaluate_literature_method_v1.py`；
- fixed-budget frontier：`collect_fixed_budget_frontier_v1.py` 与 `summarize_fixed_budget_frontier_v1.py`。

严格复现边界和 common-scope 改动见 [docs/BASELINES.md](docs/BASELINES.md)。

## 校准

仓库同时保留：历史经验计数预算 `B`、`B + calibration accuracy floor`、固定共享阈值、coverage matching，以及 problem-level Bonferroni CP、fixed-sequence Learn-Then-Test、first-failure conformal、trajectory-envelope LTT。当前校准研究统一以 calibration token reduction 选取可行阈值，不用 wall time，也不看 OOD test 选择策略。详见 [docs/CALIBRATION.md](docs/CALIBRATION.md)。

## 本地验证

```bash
python -m compileall -q src scripts tests
pytest -q
python scripts/validate_release.py
python scripts/test_deepseek7b_probe_pipeline_v1.py \
  --config configs/deepseek7b_main_v1.yaml
```

最后一项是合成缓存上的端到端 probe contract smoke test，不下载或加载 7B 模型。完整验证范围和限制见 [VALIDATION.md](VALIDATION.md)。

## 目录

```text
configs/   冻结科学协议（路径已可移植化）
docs/      方法、协议、校准和基线复现说明
scripts/   数据准备、采集、训练、评测、审计、汇总
src/       共享模型加载、标签、probe、policy replay 实现
splits/    已冻结的历史 Qwen 问题级 split 清单
tests/     单元测试与回归测试
```

结果、缓存和日志默认写入被 `.gitignore` 排除的 `results/`、`outputs/`、`logs/`。
