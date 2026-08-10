# Qwen3 推理的修正潜力停止方法

这是论文实验的设备无关参考实现：冻结 `Qwen/Qwen3-4B`，在句子级检查点上使用第 20 层隐藏动态预测“继续推理后仍可能纠正”的风险，并在风险足够低时停止自由推理、只生成一次最终答案。

仓库覆盖当前单种子旧经验协议：完整 GSM8K official test 与覆盖 57 学科的 MMLU-1k，包含固定数据划分、Dense/Direct/强制作答公共缓存、四种停止目标、轨迹最弱点保护、101 分位点经验风险校准、固定预算基线、回放延迟、10,000 次配对 bootstrap、表格和中英文总结。

> 本发布包故意不包含特定服务器的共享任务队列、A100/2080 Ti 生产者—消费者调度、GPU 守护进程或 NAS 路径。单卡可以直接运行；多卡使用静态分片参数启动多个进程。科学协议、缓存格式和输出不随调度方式改变。

## 1. 方法摘要

对 Dense 推理的检查点 `t`：

- `c_t`：在 `t` 截断并追加统一最终答案后缀后，强制作答是否正确；
- `f`：完整 Dense 推理的最终答案是否正确；
- `W→C`：`c_t=0, f=1`，提前停止会丢失一个原本正确的答案；
- `C→W`：`c_t=1, f=0`；
- `W→W`：`c_t=0, f=0`；
- `C→C`：`c_t=1, f=1`。

主标签与在线决策为：

```text
y_t = 1[当前答案错误且 Dense 最终答案正确]
q_t = P(W→C | z_t)

q_t > tau   -> 继续推理
q_t <= tau  -> 停止推理，追加最终答案后缀，只生成一次答案
```

主特征固定使用第 20 层：

```text
z_t = [h_t, delta_h_t, t, log(1+t), delta_t,
       entropy_tail8, norm(delta_h_t), cosine(h_t, delta_h_t)]
delta_h_t = h_t - h_previous
```

`h_t` 与 `delta_h_t` 各 2,560 维，另有 6 个标量特征，总宽度为 5,126。`entropy_tail8` 是最近 8 个推理 token 的平均 top-20 下一 token 熵，不是强制作答熵。

MLP 与训练配置：

```text
Linear(5126, 384) -> LayerNorm -> GELU -> Dropout(0.15)
-> Linear(384, 96) -> GELU -> Dropout(0.10) -> Linear(96, 1)

AdamW, lr=2e-4, weight_decay=1e-3
每批 24 条轨迹，梯度裁剪=2.0
最多 24 个训练轮次，耐心值=6
L = L_point + L_traj，轨迹软最小值 beta=0.5
```

基础模型始终处于 `eval()` 模式，所有参数均为 `requires_grad=False`；只训练小型探针。详细定义见 [方法规范](docs/METHOD.md)。

## 2. 固定实验协议

| 项目 | 固定值 |
|---|---|
| 随机种子 | `20260803` |
| 模型 | `Qwen/Qwen3-4B`，修订版本 `1cfa9a7208912126459214e8b04321603b3df60c` |
| 数据类型 | FP16，不量化 |
| 注意力后端 | PyTorch SDPA |
| 采样参数 | 温度 0.6、top-p 0.95、top-k 20 |
| Dense 上限 | 4,096 个新 token |
| 强制作答和 Direct 上限 | 16 个新 token |
| 句子检查点 | 从 64 开始，最晚 768，相邻至少 8 个 token；边界为换行或 `. ! ? ;` |
| 缓存的固定位置 | 64、96、128、192、256、384、512、768 |
| 主表固定预算基线 | 64、96、128、192、256 |
| GSM8K | 官方 train 中固定 1,000 个探针训练样本和 500 个策略校准样本；完整 official test 1,319 题 |
| MMLU | 非 test 数据中按 57 学科分层固定 1,000 个探针训练样本和 500 个策略校准样本；official test 中按学科平衡固定 1,000 题，每科 17 或 18 题 |
| MMLU 提示 | 每个学科使用 dev 中的标准 5-shot 示例 |

每个生成任务的随机种子只由下列键决定，不依赖启动时间、GPU、工作进程或分片：

```text
(global_seed, dataset, split, sample_id, checkpoint)
```

当前结果配置文件：

- `configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml`
- `configs/final_paper_legacy_v4_existing_fp16_mmlu.yaml`

这里的 MMLU 结果必须写作 **MMLU-1k**，不能写作完整 MMLU test。选择过程只依赖固定 seed、学科和 sample ID，不读取模型输出、正确率、轨迹长度或延迟。
采用 1,000 题的同类工作依据、统计限制与报告边界见 [MMLU-1k 协议说明](docs/MMLU_1K_PROTOCOL.md)。

## 3. 安装

要求 Python 3.10+、Linux、CUDA GPU。4B 模型 FP16 单副本需要约 8 GB 权重显存，长上下文还需要 KV cache 和临时空间；建议至少 16 GB，11 GB 卡上的少量长前缀可能需要转到更大显存设备。

```bash
git clone <你的_GitHub_仓库地址>
cd correction-potential-stopping

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 先按本机 CUDA 和驱动版本安装 PyTorch 2.6.x，再安装其余依赖。
pip install -r requirements.txt
```

下载固定修订版本；如果文件已经位于 Hugging Face 缓存中，`hf` 会直接复用：

```bash
hf download Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --local-dir models/Qwen3-4B
```

先运行无需模型推理的测试：

```bash
pytest -q
python -m compileall -q src scripts tests
```

## 4. 数据准备与泄漏约束

```bash
python scripts/prepare_final_paper_data.py
python scripts/prepare_final_paper_single_seed_scope.py
python scripts/prepare_final_paper_smoke.py
```

脚本固定下载或复用以下数据集修订版本：

- `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866`
- `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`

也可以显式传入本地快照根目录：

```bash
python scripts/prepare_final_paper_data.py \
  --gsm-snapshot /path/to/gsm8k/snapshot \
  --mmlu-snapshot /path/to/mmlu/snapshot
```

注意：GSM 快照参数应指向包含 `main/` 的修订版本根目录。

第一个脚本写出不可变父数据与完整 ID 清单；第二个脚本验证仓库内冻结 ID，并写出本轮选择层：

```text
data/final_paper_replay_v2/{gsm8k,mmlu}/
results/final_paper_replay_v2/splits/{gsm8k,mmlu}_split.json
splits/legacy_empirical_v4_train1000_cal500_mmlu1k/
├── scope_manifest.json
├── audit_selection.json
└── {gsm8k,mmlu}_{probe_train,calibration,heldout}_ids.json
```

仓库的 `splits/` 同时保存本次论文协议实际使用的固定清单。重新运行选择层必须得到相同指纹和 ID；任何路由或排序漂移都应使检查失败，而不能静默产生另一套样本。服务器本轮运行清单已与仓库清单逐 ID 比较，六个 dataset/split 组合完全一致。

MMLU 固定快照中的 `auxiliary_train.subject` 为空。父协议使用 dev/validation 问题的 TF-IDF 中心对去重后的辅助训练问题执行确定性学科路由；本轮从父清单冻结为按 57 学科分层的 1,000 个 probe 和 500 个 calibration。MMLU-1k 在每科 official test 内固定选择 17 或 18 题。六个精确 ID 清单随仓库发布，重新运行选择脚本必须逐项一致。MMLU calibration 来自 `auxiliary_train`，相对 official test 存在明确分布偏移，结果必须标记为 distribution-shift；test 标签从不参与路由、训练或阈值选择。

## 5. 公共缓存格式

缓存分为三层，均采用只追加方式；不兼容文件不会被覆盖：

```text
cache/<数据集>/
├── dense/<划分>/sample_<id>.pt       # Dense token、特征和检查点前缀
├── branches/<划分>/<id>/             # Direct 和每个强制作答分支
└── merged/<划分>/sample_<id>.pt       # 不可变公共缓存视图
```

唯一语义键为 `(dataset, split, sample_id, checkpoint)`。所有自适应方法只改变训练标签或轨迹损失；它们读取同一合并缓存，绝不重新生成答案。

`--resume` 仅复用同时满足以下条件的文件：状态完整、sample ID 一致，并且模型元数据、数据类型、提示、解码参数、检查点协议和数据划分清单的联合指纹一致。不匹配文件会被保留并报错。

## 6. 小规模完整链路检查

先使用第 4 节产生的选择文件。GSM8K 每个划分使用 20 题；MMLU 每个划分的每个学科使用 1 题，共 57 题。

下面以 GSM8K calibration 为例；需要对 `probe_train calibration heldout` 以及 `gsm8k mmlu` 重复执行。运行 MMLU 时更换配置、数据清单、缓存根目录和对应 ID 文件。

```bash
python scripts/collect_final_paper_dense_cache.py \
  --dataset gsm8k \
  --config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --split-manifest results/final_paper_replay_v2/splits/gsm8k_split.json \
  --split calibration \
  --cache-root results/final_paper_replay_v2/cache/gsm8k \
  --sample-ids results/final_paper_replay_v2/selections/gsm8k_calibration_smoke_ids.json \
  --gpu 0 --measure-timing --resume

python scripts/collect_final_paper_branch_cache.py \
  --dataset gsm8k \
  --config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --split calibration \
  --cache-root results/final_paper_replay_v2/cache/gsm8k \
  --gpu 0 --resume

python scripts/merge_final_paper_replay_cache.py \
  --cache-root results/final_paper_replay_v2/cache/gsm8k \
  --split calibration --resume
```

两个数据集的三个划分都完成后执行小规模缓存审计：

```bash
python scripts/audit_final_paper_replay_cache.py \
  --cache-base results/final_paper_replay_v2/cache \
  --mode smoke \
  --selection results/final_paper_replay_v2/selections/smoke_selection.json \
  --gsm8k-config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --mmlu-config configs/final_paper_legacy_v4_existing_fp16_mmlu.yaml \
  --splits-root results/final_paper_replay_v2/splits \
  --output results/final_paper_replay_v2/SMOKE_CACHE_AUDIT.json
```

只有 `status=PASS` 时才继续正式收集。已完成的小规模样本与正式配置完全一致，后续 `--resume` 会直接复用。

## 7. 正式公共缓存：单卡与多卡

### 单卡

对两个数据集和三个划分依次运行 Dense，再运行分支采集，最后合并。每次 Dense 采集都必须传入本轮固定 ID 文件；以下是完整 GSM8K held-out 的示例：

```bash
python scripts/collect_final_paper_dense_cache.py \
  --dataset gsm8k \
  --config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --split-manifest splits/gsm8k_split.json \
  --split heldout \
  --cache-root results/legacy_empirical_v4/cache/gsm8k \
  --sample-ids splits/legacy_empirical_v4_train1000_cal500_mmlu1k/gsm8k_heldout_ids.json \
  --gpu 0 --resume

python scripts/collect_final_paper_branch_cache.py \
  --dataset gsm8k \
  --config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --split heldout \
  --cache-root results/legacy_empirical_v4/cache/gsm8k \
  --gpu 0 --resume

python scripts/merge_final_paper_replay_cache.py \
  --cache-root results/legacy_empirical_v4/cache/gsm8k \
  --split heldout --resume
```

对 `probe_train` 和 `calibration` 重复时替换 split 与对应 ID 文件。MMLU 使用 `configs/final_paper_legacy_v4_existing_fp16_mmlu.yaml` 和冻结目录中的 `mmlu_<split>_ids.json`。分支采集读取已经存在的 Dense 文件，因此不再接收 ID 文件。

### 多卡静态分片

任意数量、任意型号 GPU 均使用相同命令。以 2 张卡为例，同时启动：

```bash
python scripts/collect_final_paper_dense_cache.py ... \
  --gpu 0 --num-shards 2 --shard-index 0 --resume

python scripts/collect_final_paper_dense_cache.py ... \
  --gpu 1 --num-shards 2 --shard-index 1 --resume
```

分支采集同样加入 `--num-shards 2 --shard-index 0/1`。所有 Dense 分片完成后再启动分支分片，避免任务集合在进程启动后改变。不同 GPU 的性能只影响完成时间，不影响样本输出。

如果分支任务在小显存卡上出现显存不足，记录中的样本和检查点会保持缺失；使用更大显存 GPU 和相同参数重新运行即可。不要量化、缩短前缀或修改 token 上限。

正式语义缓存完成后：

```bash
python scripts/audit_final_paper_replay_cache.py \
  --cache-base results/legacy_empirical_v4/cache \
  --mode formal \
  --selection results/legacy_empirical_v4/splits/audit_selection.json \
  --gsm8k-config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --mmlu-config configs/final_paper_legacy_v4_existing_fp16_mmlu.yaml \
  --splits-root splits \
  --output results/legacy_empirical_v4/CACHE_INTEGRITY_AND_LEAKAGE_AUDIT.json
```

审计包括样本完整性、数据划分泄漏、57 学科覆盖、配置指纹、token 长度、hidden 中的 NaN/Inf、记录与向量对齐、重复或缺失检查点、Direct 和强制作答分支，以及五元组随机种子。

## 8. 时间校准与回放延迟

时间校准和策略校准是两个不同概念：

- **时间校准**：只拟合目标部署 GPU 的上下文长度—生成成本；不训练停止器，也不选择停止阈值。
- **策略校准**：使用 500 个 calibration 问题按历史经验 lost-correct 绝对预算选择阈值；不拟合硬件速度。

准备脚本已固定选择 200 个 GSM8K probe-train 问题，以及 MMLU 每个学科 8 个、共 456 个 probe-train 问题。使用目标部署 GPU、单请求、预热、FP16、SDPA 和 `--measure-timing`，写入独立缓存；不能使用分支工作进程的时间：

```bash
python scripts/collect_final_paper_dense_cache.py \
  --dataset gsm8k \
  --config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --split-manifest results/final_paper_replay_v2/splits/gsm8k_split.json \
  --split probe_train \
  --cache-root results/final_paper_replay_v2/timing_cache/gsm8k \
  --sample-ids results/final_paper_replay_v2/selections/gsm8k_probe_train_timing_ids.json \
  --gpu 0 --measure-timing --resume

python scripts/collect_final_paper_dense_cache.py \
  --dataset mmlu \
  --config configs/final_paper_legacy_v4_existing_fp16_mmlu.yaml \
  --split-manifest results/final_paper_replay_v2/splits/mmlu_split.json \
  --split probe_train \
  --cache-root results/final_paper_replay_v2/timing_cache/mmlu \
  --sample-ids results/final_paper_replay_v2/selections/mmlu_probe_train_timing_ids.json \
  --gpu 0 --measure-timing --resume
```

拟合脚本按照 sample ID 固定划分 80% 拟合集和 20% 验证集；held-out 不会被扫描：

```bash
python scripts/fit_final_paper_a100_cost_model.py \
  --cache-root gsm8k=results/final_paper_replay_v2/timing_cache/gsm8k \
  --cache-root mmlu=results/final_paper_replay_v2/timing_cache/mmlu \
  --output results/final_paper_replay_v2/single_request_cost_model.json
```

冻结条件为 Dense 总时间与检查点前缀成本的验证集 MAPE 均不超过 5%，p95 相对误差均不超过 10%。不达标时按固定 200 条增补时间样本；不要使用 held-out 结果选择模型，也不要为了微小误差无限扩充。

测量停止器和检查点模块开销；使用初始化权重即可：

```bash
python scripts/benchmark_final_paper_stopper.py \
  --gpu 0 \
  --output results/final_paper_replay_v2/stopper_overhead.json
```

使用冻结成本模型创建派生回放视图。`--probe-overhead-ms` 使用上一步同步后端到端时间的平均值；成本文件只有一个设备时无需指定 `--cost-device`：

```bash
python scripts/materialize_final_paper_replay_view.py \
  --dataset gsm8k \
  --config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --cache-root results/legacy_empirical_v4/cache/gsm8k \
  --cost-model results/final_paper_replay_v2/single_request_cost_model.json \
  --probe-overhead-ms <平均毫秒数> \
  --output-root results/legacy_empirical_v4/replay/gsm8k \
  --resume
```

对 MMLU 重复执行。如果成本文件包含多个设备，使用 `--cost-device '<准确的设备键>'` 冻结一个目标部署设备。最终表述必须使用成本 JSON 中的标签，例如：

```text
A100-SXM4-80GB 单请求回放估计延迟
```

不得写成完整在线策略的实测端到端时间。

## 9. 基线与自适应探针

先计算 Dense、Direct 和固定预算基线：

```bash
python scripts/evaluate_legacy_empirical_baselines_v4.py \
  --dataset gsm8k \
  --config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --dense-root results/legacy_empirical_v4/replay/gsm8k \
  --checkpoint-root results/legacy_empirical_v4/replay/gsm8k \
  --output results/legacy_empirical_v4/gsm8k/baselines \
  --resume
```

五个探针运行共享同一特征、标准化器、数据划分、MLP 和优化器，只改变预测目标或轨迹损失：

```bash
# 受控预测目标：仅使用检查点 BCE
python scripts/train_legacy_empirical_probe_v4.py \
  --dataset gsm8k --config configs/final_paper_legacy_v4_existing_fp16_gsm8k.yaml \
  --raw-root results/legacy_empirical_v4/replay/gsm8k \
  --output results/legacy_empirical_v4/gsm8k/probes/correctness \
  --method correctness --loss bce --schedule sentence --layer 20 \
  --feature-kind full --seed 0 --gpu 0 --resume

python scripts/train_legacy_empirical_probe_v4.py ... \
  --output results/legacy_empirical_v4/gsm8k/probes/consistency \
  --method consistency --loss bce

python scripts/train_legacy_empirical_probe_v4.py ... \
  --output results/legacy_empirical_v4/gsm8k/probes/last_switch \
  --method last_switch --loss bce

# 损失函数消融
python scripts/train_legacy_empirical_probe_v4.py ... \
  --output results/legacy_empirical_v4/gsm8k/probes/correction_bce \
  --method correction --loss bce

# 主方法
python scripts/train_legacy_empirical_probe_v4.py ... \
  --output results/legacy_empirical_v4/gsm8k/probes/correction_trajectory \
  --method correction --loss bce_traj
```

将 `gsm8k` 换为 `mmlu` 后完整重复。正确性、一致性和最后切换是同一框架中的受控停止目标基线，不是对其他论文全部训练细节的官方复现。

`StandardScaler` 只在 probe-train 内部固定的 80% 拟合问题上训练，20% 仅用于 epoch 选择。训练完成后在 500 个 calibration 问题上扫描 101 个 score 分位点，并加入完全不停止的 sentinel。历史主协议使用 calibration lost-correct 的绝对数量预算 `B={0,1,2,4,10}`；Strict、Balanced、Aggressive 分别是 `B=1,2,4`。它们是经验事件预算，不是二项分布置信上界。另行报告 30%–90% calibration coverage-targeted 工作点；held-out 从不选择阈值。

## 10. 汇总、配对 bootstrap 与报告

两个数据集的基线和探针目录完成后：

```bash
python scripts/compile_legacy_empirical_results_v4.py \
  --run-root results/legacy_empirical_v4 \
  --cache-audit results/legacy_empirical_v4/CACHE_INTEGRITY_AND_LEAKAGE_AUDIT.json \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260803
```

编译器会验证所有方法使用相同 held-out sample IDs，产生：

```text
results/legacy_empirical_v4/
├── tables/
│   ├── main_results.csv
│   ├── historical_empirical_B.csv
│   ├── coverage_targeted.csv
│   ├── target_ablation.csv
│   ├── loss_ablation.csv
│   ├── bootstrap_confidence_intervals.csv
│   ├── paired_comparisons.csv
│   ├── risk_frontier.csv
│   ├── mmlu_subject_results.csv
│   └── mmlu_category_results.csv
├── figures/
├── FINAL_EXPERIMENT_SUMMARY_ZH.md
├── FINAL_EXPERIMENT_SUMMARY_EN.md
└── pipeline.complete
```

每次 bootstrap 都对全部方法重采样相同的问题 ID，报告准确率、token 减少率、平均和 p95 回放延迟减少率、lost-correct 风险与覆盖率的 95% 置信区间，以及主方法相对各基线的配对差值置信区间。不使用 `C→W` 抵消 `W→C` 风险。

## 11. 环境审计

数据清单与模型就绪后，在采集前运行：

```bash
python scripts/audit_environment.py \
  --output results/final_paper_replay_v2/ENVIRONMENT_AUDIT.json
```

它记录 Python、torch、Transformers、CUDA、驱动、GPU 型号和显存、模型权重元数据指纹、数据修订版本和样本数、数据类型、注意力后端、解码参数和随机种子。

## 12. 关键实现文件

- `src/final_paper_protocol.py`：MMLU 57 学科及类别、5-shot 提示、选择题解析器和检查点定义；
- `src/final_paper_cache.py`：五元组随机种子、配置指纹、公共缓存路径与句子边界；
- `src/qwen3_reasoning.py`：冻结 Qwen3、采样、Dense 轨迹与 CUDA 计时；
- `src/final_paper_inference.py`：提示、GSM/MMLU 判分、hidden hook 和强制作答；
- `src/final_paper_probe.py`：通用 5,126 维特征、MLP 和标签实现；
- `src/legacy_empirical_probe_v4.py`：本轮加权 Correction loss、trajectory soft-min、历史经验 B 校准和 first-hit 回放；
- `scripts/train_legacy_empirical_probe_v4.py`：统一训练 Correctness、Consistency、Last-switch、Correction BCE 与完整方法；
- `scripts/compile_legacy_empirical_results_v4.py`：历史 B/coverage 表、配对 bootstrap、MMLU 分解和中英文总结；
- `src/final_paper_online.py`：可选的不依赖答案探针的句子级在线参考实现；论文主表不依赖完整策略的在线计时。

## 13. 可复现性规则

1. 修改固定的模型或数据集修订版本、随机种子、提示、解码参数或解析器后，不得混用旧缓存。
2. official test 不得参与探针训练、标准化器拟合、早停、阈值选择或成本模型拟合。
3. 强制作答只用于构造离线标签；自适应回放不生成新的检查点答案。
4. 缺少字段时应补齐公共缓存，不能只为某一种方法排除样本。
5. 不覆盖不兼容的产物；应写入新目录并保留诊断证据。
6. `CACHE_AUDIT.json` 不是 `PASS` 时不得编译论文表。
7. 报告中明确区分校准集选择的结果、held-out 描述性诊断和单请求回放估计。

完整核对表见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 14. 许可证与引用

本代码包不包含 Qwen3 权重或 GSM8K/MMLU 数据。使用者需遵守模型和数据集各自许可证。发布到 GitHub 前，请由项目作者选择并添加适合的代码许可证与正式引用信息；本整理过程没有替作者擅自指定许可证。
