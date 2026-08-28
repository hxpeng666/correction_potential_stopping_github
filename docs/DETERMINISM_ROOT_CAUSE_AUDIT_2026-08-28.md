# DeepSeek grader-comparison determinism root-cause audit

## Verdict

The earlier original-vs-forced-at-cap grader table was not a valid paired
comparison.  Its "original grader" numbers came from the cached 21 August probe,
whereas the forced-at-cap condition was newly trained by
`run_deepseek7b_forced_cap_grader_method_v1.py`.  That runner did not retrain the
original condition.  Consequently the table mixed two training executions.

This is why GSM8K correctness changed even though its row keys, features and
labels were grader-invariant.  The change is not a correctness-label effect and
must not be attributed to the grader.

## Historical artifact limitation

The cached probe was written on 21 August.  `PROTOCOL_FREEZE.json` was created on
25 August, so its source hash is a post-hoc snapshot rather than proof of the
exact source used to create that probe.  The historical artifact does not record
an initial-state hash, deterministic CUDA settings, full environment identity,
or a Git commit.  No timestamped 21 August training-source snapshot was found.
The exact unrecorded instruction/RNG state of that run is therefore not
identifiable after the fact.

The available post-hoc source snapshot reproduces the current pre-fix rerun
bitwise, but not the 21 August probe.  Relative to the historical probe, the
maximum absolute final-state difference is `0.008695580996572971` and the maximum
score difference is `0.126358300447464`.

Controlled reversions did not recover the historical artifact:

- the original supervisor's `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4` environment;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- TF32 with `float32_matmul_precision=high`;
- probe seed 42;
- the Dense-generation seed 20260820.

These tests rule out the obvious recorded settings.  They cannot distinguish an
unarchived source edit from an unrecorded RNG-consumption state, because the
historical run saved neither the actual commit nor its initial model state.

## Corrected protocol and evidence

The corrected paired runner independently trains both grader conditions from one
clean commit and aborts unless the grader-invariant correctness probes are
bitwise identical.  The completed GSM8K and MATH negative controls have exact
initial states, histories, best epochs, final tensors and per-split score tensors;
all maximum score differences are zero.

An additional GSM8K run on the second A100 is bitwise identical to the first A100
for the same quantities.  The evidence files are stored under the deterministic
grader-pair output as `NEGATIVE_CONTROL_GSM8K.json`,
`NEGATIVE_CONTROL_MATH500.json`, and `CROSS_GPU_GSM8K.json`.

Future formal v2 runs also require the committed runtime lock.  Changed code,
input hashes, package versions, CUDA/cuDNN, driver, or uncertified GPU UUIDs cause
an immediate refusal instead of a silent rerun.

The full v2 training entry point was then run once on each A100 from commit
`91adb3cbd0e1119711ba14e384cd2f591e3a0ec9`.  The independent audit
`results/deepseek7b_runtime_lock_cert_v2/BITWISE_AUDIT.json` reports
`all_exact=true`: input identity, initial state, complete training history, best
epoch, every final parameter tensor, and every probe-train/calibration/held-out
score tensor are identical.  The maximum and mean absolute score differences on
all three splits are exactly zero.  Serialized `probe.pt` files themselves are
not expected to have the same byte hash because they include run-specific
provenance metadata; the numerical model state and decision scores are compared
separately and are bitwise exact.

The stochastic Dense collector was also moved behind the same fail-closed
runtime lock.  It retains the configured sampling distribution but derives a
dedicated RNG seed from `(global_seed, problem_id)`, making worker count and
execution order irrelevant.  At commit
`efaeb5934bfe990429c251f6dbb1133be92027dc`, the real held-out sample
`gsm8k_test_00059` was collected independently on both certified A100s.  The
collection audit reports identical Dense and forced-answer token IDs, entropy
values, checkpoint positions, labels and hidden-state tensors; the scientific
payload SHA-256 is identical and the maximum hidden-state difference is zero.
Only operational fields such as wall time, timestamp, worker and GPU identity
are excluded from equality.

## Reporting consequence

The historical grader-comparison table and any table that mixes the historical
probe with a current retrain are non-canonical.  Feature, BCE-weight, and
first-hit/trajectory ablations should be reported only from the committed
deterministic rerun after its gate and result audit complete.
