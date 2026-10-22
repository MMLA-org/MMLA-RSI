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
