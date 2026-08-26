# 发布版验证

上传前执行以下设备无关验证：

```bash
python -m compileall -q src scripts tests
pytest -q
python scripts/validate_release.py
python scripts/test_deepseek7b_probe_pipeline_v1.py \
  --config configs/deepseek7b_main_v1.yaml
```

检查范围：

- 所有发布 Python 文件可编译；
- label、answer equivalence、checkpoint、feature、policy replay 与 normalized soft-min 回归测试；
- YAML/JSON 可解析、DeepSeek 路径为仓库相对路径、配置维度和关键协议不变量一致；
- 不包含权重、`.pt` 缓存、结果或日志；
- 合成 DeepSeek 3590-D 缓存可完成 probe 训练、经验 B 阈值回放与 AIME-style OOD frozen-probe 评测。

合成 smoke 不加载 7B 模型，因此验证的是发布代码拼装、训练和评测契约。完整 GPU generation 仍要求使用者提供权重、数据快照和 CUDA 环境；仓库不宣称在无权重的本地机器上复跑完整数千题采集。

本文件在每次发布时更新为实际命令、通过数量和环境版本；不要把历史通过记录当作当前 commit 的证据。

## 2026-08-26 本地发布验证

- Python 3.13.5；
- PyTorch 2.6.0、Transformers 4.53.3、NumPy 2.1.3、pandas 2.2.3、SciPy 1.15.3、scikit-learn 1.6.1；
- `compileall`：通过；
- `pytest -q`：47 passed；
- `validate_release.py`：190 个 Python、47 个 YAML、2 个 JSON 均通过语法/解析/路径/协议检查；
- 15 个核心 CLI 的 `--help` import 检查：通过；
- DeepSeek 3590-D synthetic contract：correctness 与 BCE+normalized-trajectory 训练、经验 B replay、AIME frozen-probe OOD 评测全部完成。
