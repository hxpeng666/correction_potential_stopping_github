# Qwen3-14B / DeepSeek-7B data-protocol alignment

The Qwen3-14B collection intentionally reuses the frozen DeepSeek main-v2
scientific data protocol wherever the setting is model independent.

## Matched invariants

- Dense rollout base seed: `20260820`, followed by the same SHA-256
  `(base_seed, problem_id)` derivation.
- Prompt rendering and the same two-step tokenizer call used by the DeepSeek
  collector (`apply_chat_template(..., tokenize=False)` followed by the
  tokenizer's default `add_special_tokens` behavior).
- Dense sampling: 13,000 new tokens, temperature 0.6, top-p 0.95, top-k 20.
- Forced answer: `\n</think>\n\n\\boxed{`, greedy, at most 48 new tokens.
- Paragraph checkpoints within the reasoning span, no checkpoint range filter,
  and Dense fallback when no checkpoint exists.
- Identical frozen problem IDs and problem-level GSM8K/MATH/MATH-500/AIME2024
  split files.
- Exact-13K cap hits use the current forced-at-cap grader view.

## Deliberately model-specific settings

- Model weights and tokenizer.
- Hidden width and capture layer: Qwen3-14B uses zero-based layer 20 and width
  5120; DeepSeek-7B uses zero-based layer 16 and width 3584.
- The Qwen run additionally enforces a runtime lock and cross-A100 bitwise gate.
  These strengthen reproducibility without changing the scientific data
  distribution.

## Superseded Qwen run

The prior Qwen v4 run used Dense rollout base seed `0`. It is scientifically
incompatible with the frozen DeepSeek rollout and must not contribute artifacts
to the aligned collection. Its output is retained as provenance evidence only.
