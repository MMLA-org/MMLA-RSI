# Integrating a model and a task

A model exposes `rho` (`LearningResidual`), an HF-compatible `mixer.config` and
`mixer.get_input_embeddings()`, a dataclass `profile` containing `local_tokens`
and `max_new_tokens`, `substrate_identity`, and
`forward_batch(ids, mask, fast_states=[state], memory=None, logit_positions=...)`.
The result is batch × requested-position × vocabulary logits. Keep the model in
eval mode, freeze its shared parameters and preserve gradients through the input.
The initial residual uses nonzero A and zero B; reset returns to the role's rho,
not an untrained zero state. Task and proposal roles require distinct rho/state.

`CausalLarcAdapter` accepts an already loaded causal LM, an `AdapterProfile`, a
seed and explicit model provenance. It does not move or load the backbone for you.
Its resident reader identity includes actual parameter bytes. Slow changes require
new reader bindings. Concurrent writes, raw-storage mutation and `.data` mutation
are unsupported. This minimal adapter uses full recomputation, one episode per
call and explicit masks/positions; no KV reuse is implied.

`ProposalFactory(model, tokenizer, tokenizer_identity=..., language=...,
memory=..., attempts=..., update_bounds=...)` maps grammar symbols to distinct
single-token characters. Each character must round-trip through the real tokenizer.
The factory provides a complete immutable interface header, then creates fresh
private Phi/M for each generation. Allocate enough context for that header, the
entire memory and the bounded output. Unsupported tokenizers fail rather than
silently changing the proposal distribution.

For a task, supply `TaskSampling`, `TaskPolicy`, a `TaskLearner` and a backend
implementing `feedback(action, support, opportunity=..., meter=...)` and
`score(action, final, meter=...)`. `TaskPayloads` binds each `Case.payload_sha256`
to its actual request/support/final payload. The feedback path must use only
legally arrived support. Keep final labels out of prompts, rule features, caches
and proposal learning. Put hard timeouts and untrusted-tool isolation in the host.

`Evaluator` aggregates complete task results and charges an explicit `Meter`.
`Controller` receives the language, parent, `RunConfig`, proposal factory,
evaluator, development cases, per-generation audit groups and a `Journal`.
Decide the task loss, resource units, margins, independence rules and finite
budgets before evaluation. Read the callable signatures and tiny example for an
executable assembly. Do not use the demo endpoint for training or model selection.

## Experimental LARC CUDA operators

[MMLA-Kernels](https://github.com/MMLA-org/MMLA-Kernels) provides an experimental
rank-4 CUDA forward/backward. Its FP32-factor/BF16-activation residual arithmetic
matches `LearningResidual.inject`: cast the low-rank branch before adding
the residual, and retain the corresponding cast in the input gradient. The
CUDA implementation is first-order only. Compilation for sm89/90/100/103/120 plus compute100 PTX is complete;
GPU numerical/performance qualification is pending. This RSI adapter continues
to use PyTorch and does not automatically dispatch to the experimental kernel.

A future backend integration must retain per-role factor ownership, action
probabilities, feedback gradients, commit/reset state and exact recovery. It
must invalidate dependent KV caches after changes to Phi/M/reader identity.
Operator acceleration neither changes the rule-admission criterion nor supplies
evidence of recursive improvement. Include proposal, feedback, evaluation and
checkpoint costs when measuring an RSI process.

The kernel [architecture matrix](https://github.com/MMLA-org/MMLA-Kernels/blob/main/docs/ARCHITECTURES.md)
tracks Ada (`sm89`), Hopper (`sm90`), B200/GB200 (`sm100`),
B300/GB300 (`sm103`) and RTX PRO6000 (`sm120`)
separately. Native compilation, generic PTX compatibility, numerical checks and
end-to-end throughput are distinct evidence; qualify the actual deployment GPU.
