# 发布内容清单

## 收录

- Qwen3-4B 与 DeepSeek-R1-Distill-Qwen-7B 的冻结科学配置；
- GSM8K、MMLU/MMLU-Pro、Hendrycks MATH、MATH-500、AIME 2024 的问题级准备与验证逻辑；
- Dense rollout、paragraph/sentence/fixed/prefix/cue/hybrid checkpoint、greedy forced-answer 采集；
- correctness、consistency、last-switch、correction BCE、normalized trajectory 五组 probe；
- layer、feature、checkpoint schedule、suffix、fixed-budget 和 dynamic stopping 消融；
- LTS、LYNX、self-verification、Thought Calibration 的 native 与 paragraph 复现；
- 经验 B、accuracy-floor、coverage、formal LTT/conformal 校准研究；
- 13K 到 selective-32K 的精确增量迁移及审计；
- fixed relative-budget frontier、OOD 复用评测、bootstrap、审计和汇总；
- 单元测试、合成端到端 smoke test、协议注册表和复现文档。

## 排除

- 模型权重与 Hugging Face cache；
- 原始题目副本、生成 trajectory、hidden tensor、probe 权重；
- 实验结果、图表、日志、PID、进程状态与 incident 证据；
- cool100/NAS/Conda 绝对路径；
- GPU 抢占、显存探测、动态 collector allocator、supervisor、tmux/nohup 等机器调度代码；
- 临时备份、`.pre_*` 文件和不完整缓存。

这些排除项不改变模型、数据、标签、checkpoint、loss、calibration 或 metric 语义。
