# 脚本索引

为保持服务器实验 artifact 与命令行兼容，脚本文件名未重排到子目录；按功能使用前缀组织：

| 前缀 | 用途 |
|---|---|
| `prepare_*` | 冻结问题级 split、数据 manifest 或 replay scope |
| `collect_*` | Dense、checkpoint、forced-answer、fixed-budget 数据采集 |
| `train_*` | 五组 probe、trajectory、dynamic/utility 消融训练 |
| `evaluate_*` | frozen threshold、OOD transfer、baseline replay |
| `run_deepseek7b_calibration_*` | 经验与 formal token-objective 校准研究 |
| `migrate_deepseek7b_*` | 13K 缓存迁移和 selective-32K 精确增量续跑 |
| `audit_*` / `verify_*` | 协议指纹、数量、标签、结果逐值审计 |
| `summarize_*` / `compile_*` | 将已有 artifact 汇总成表；不包含预生成结果 |

## DeepSeek 当前主链

1. `prepare_deepseek7b_data_v1.py`
2. `freeze_deepseek7b_protocol_v1.py`
3. `collect_deepseek7b_paragraph_v1.py`
4. `repair_deepseek7b_numeric_labels_v2.py`
5. `audit_deepseek7b_collection_v1.py`
6. `train_deepseek7b_ablation_v1.py`
7. `evaluate_deepseek7b_ood_v2.py`
8. `summarize_deepseek7b_results_v1.py`
9. `audit_deepseek7b_completion_v1.py`

正式 `bce_traj` 必须使用 `--trajectory-aggregation normalized_softmin --trajectory-beta 0.5 --trajectory-weight 1`；发布版训练器默认也已设为 normalized。

## Qwen checkpoint/目标链

1. `collect_final_paper_dense.py`
2. `collect_final_paper_checkpoints.py` 或 `collect_full_checkpoint_schedule_v1.py`
3. `collect_greedy_forced_answer_v1.py`
4. `train_legacy_empirical_probe_v4.py` / `train_controlled_label_normalized_trajectory_v1.py`
5. 对应 `evaluate_*`、`audit_*` 和 `summarize_*`

## 相关工作

`collect_literature_method_data_v1.py` + `train_evaluate_literature_method_v1.py` 是 LTS、LYNX、self-verification 与 Thought Calibration 的统一入口。论文原生 checkpoint 和 paragraph common-checkpoint 必须分别运行并分别标注。

服务器上的动态 collector allocator、supervisor、GPU 抢占监控和后台 worker shell 脚本没有发布；这些只负责吞吐，不属于科学协议。
