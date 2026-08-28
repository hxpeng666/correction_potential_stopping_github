# Experiment-to-commit reproducibility protocol

Every formal experiment must be attributable to one immutable Git commit.  A
result directory is valid only when its `RUN_MANIFEST.json` and probe artifacts
record all of the following:

1. the full Git commit SHA, branch and remote URL;
2. `dirty: false` for the experiment worktree;
3. the exact command and configuration-file SHA-256 values;
4. row-key, feature and target hashes for every training/calibration/test split;
5. Python, PyTorch, CUDA, cuDNN, GPU UUID and driver versions;
6. the deterministic protocol, seed, initial-model hash, final-model hash and
   per-split score hashes.

Formal runners refuse to start from a dirty worktree or to resume an existing
output directory with a different invocation fingerprint.  Experiments that do
not yet have a dedicated runner must be launched through
`scripts/run_committed_experiment_v1.py`.

## Frozen deterministic settings

- seed: `0` for Python, NumPy and PyTorch;
- `PYTHONHASHSEED=0`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`;
- `torch.use_deterministic_algorithms(True)`;
- cuDNN benchmark disabled and deterministic mode enabled;
- TF32 disabled;
- AdamW fused/foreach kernels disabled;
- one probe-training process per GPU.

The same settings are mandatory for formal Dense/checkpoint collection. Dense
decoding intentionally retains the scientific sampling protocol
(`temperature=0.6`, `top_p=0.95`, `top_k=20`); reproducibility does not mean
silently changing it to greedy. Instead, each problem receives a dedicated
seed derived by SHA-256 from `(global_seed, problem_id)`, so worker count,
sharding and scheduling order cannot change its random stream. The collection
artifact records that derived seed and the locked code/runtime identity.

Low-level DeepSeek collection and probe-training entry points fail closed when
`reproducibility.runtime_lock` is absent. `--allow-unlocked-legacy` is an
explicit escape hatch for examining historical protocols only; such outputs
are non-formal and must not be mixed with locked results.

## Runtime lock (required for formal v2 runs)

Recording an environment is not sufficient: a later run could record a changed
environment and still look superficially valid.  Formal v2 runners therefore
require `reproducibility.runtime_lock` in their configuration and compare the
live process against that committed JSON lock before loading training data.

The lock fixes the complete Python build string, OS/glibc identity,
scientific package versions, CUDA runtime, cuDNN, NVIDIA driver, GPU model,
memory size and compute capability.  GPU UUID is an explicit allow-list of
devices that have passed the bitwise cross-device gate.  Any mismatch aborts the
run.  Updating a driver or package requires a new lock, Git commit, output name,
and reproducibility certification; it cannot silently reuse an older result.

Bitwise reproducibility is certified only inside the committed runtime lock.
No PyTorch program can promise identical floating-point bits across arbitrary
future hardware and library versions.  Outside the lock the correct behavior is
therefore refusal, not a tolerance-based claim.

## Required reproducibility gate

Before comparing two graders or two label definitions, train an invariant
negative control independently in both data views.  For the current grader
study this is the correctness target: its row keys, features and labels must be
identical.  The two runs must have exact-equal initial weights, training history,
best epoch, final state tensors and score tensors.  The experiment aborts before
any scientific comparison if this gate fails.

Calibration results are not part of that invariant: changing the Dense final
grader can legitimately change calibration risk labels and policy metrics even
when the correctness probe is identical.

## Historical results

Results created without a recorded clean commit, input hashes and deterministic
environment are retained as historical evidence, not as the canonical table.
They must not be mixed with committed deterministic reruns in a paired
comparison.  The canonical result is the rerun whose manifest passes this
protocol.

## Run naming

Use a stable output name such as
`<model>_<study>_<protocol>_vN`.  Never overwrite an earlier directory.  The
result manifest is the authoritative mapping from that run name to its Git
commit; optional Git tags may use `experiment/<run-name>` after completion.

## Equality scope

Scientific equality is bitwise for generated token IDs, entropy values,
checkpoint positions, hidden states, forced-answer token IDs, labels, probe
parameters and scores. Operational values such as timestamps, host names,
logical GPU indices and measured wall time are recorded for diagnostics but are
excluded from scientific equality because they are expected to differ.
