# MMLA-RSI

**Recursive adaptation through RTT and RTMA.** MMLA-RSI is a Python reference
implementation of the two recursive routes in
[MMLA-RSI: Recursive Self-Improvement through RTT and RTMA](https://arxiv.org/abs/2610.20294v1).

The current adaptation rule takes part in generating its successor. An accepted
successor takes over that same role in the next generation:

- **RTT**: the current updater learns the proposal policy that generates successor
  updater programs. Its bounded expressions control step size, clipping and momentum.
- **RTMA**: the current memory manager chooses which evaluation records the proposal
  policy can read. A successor manager takes over that retention/replacement role.
- **Joint**: a rule pair runs both paths, with separate state and commit semantics.

These are executable state transitions, grammar-constrained model proposals,
feedback gradients, selection, independent admission, and version replacement.
The package does not ship pretrained weights or a measured beneficial recursive
chain. The tiny examples exercise the implementation using random models and
explicitly synthetic task endpoints.

## Install

Python 3.10+ on Linux/POSIX. Use a virtual environment; sudo is unnecessary.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[examples]'
```

PyTorch, NumPy and safetensors are the core dependencies. Transformers is needed
only for the supplied HF example adapter tests and tiny demos. Model loading is
explicit: importing this package does not download or run a model.

## Start with the mechanisms

```sh
python examples/rules_and_memory.py
python examples/tiny_recursion.py --mode joint --output runs/joint-demo
python -m unittest discover -s tests -v
```

The first example needs no pretrained model. The second creates a tiny random
causal LM on CPU and runs proposal, feedback, selection and admission. Its task
loss is a **handwritten control-flow fixture**, printed as such in the output;
accepting a fixture candidate is not evidence of useful RSI. An output directory
is single-run state and must be new. Interrupted runs can restore only a complete,
verified quiet boundary, as described in [recovery](docs/RECOVERY.md).

## What is included

| Component | Interface |
|---|---|
| Bounded prefix language, typed RTT/RTMA/joint rules | `Language`, `parse_rule` |
| Slow LARC initialization and private fast factors/RNG/momentum | `LearningResidual`, `FastState` |
| Authoritative evidence, complete-row writes, NULL and eviction | `EvidenceMemory` |
| Grammar-masked proposals and score-function replay | `ModelProposer`, `ProposalFactory` |
| Parent-rule application to the proposal learner | `ProposalLearner` |
| Full-vocabulary task learner and feedback/score backend | `TaskPolicy`, `TaskLearner`, `TaskExecutor` |
| Development lock, paired admission and next-generation handoff | `Controller`, `RunConfig` |
| Safetensors/JSON quiet checkpoints and versioned request routing | `Journal`, `ParentRegistry` |
| HF causal-LM input-LARC projection, with full recomputation | `CausalLarcAdapter` |

The adapter preserves input gradients through a frozen model. It deliberately
supports one isolated episode per call. Memory is serialized into each fresh
prompt; an evicted record cannot remain in an old KV cache. Other mixers can
implement the same [model and task interfaces](docs/INTEGRATION.md).

## Execution semantics

Phi is task-local fast policy state, and M is authoritative memory. Their
initializations, optimizer state and RNG remain distinct between roles and
problems. Joint adaptation commits the numerical update before the memory action;
a memory failure does not silently undo an earlier numerical commit. Invalid
payloads fail before action scoring; a valid payload with no writable slot may
legitimately select NULL.

Development data select one candidate; separately bound audit groups decide
whether it replaces the parent. Group provenance and statistical independence
must come from the application, not from fresh IDs. The controller records
structural eligibility separately from functional participation. A claim of a
beneficial recursive chain also requires the declared causal and task evidence.
See [algorithm semantics](docs/SEMANTICS.md).

## Related projects and papers

- [MMLA-PostTraining](https://github.com/MMLA-org/MMLA-PostTraining): slow initialization and post-training tools.
- [MMLA-PreTraining](https://github.com/MMLA-org/MMLA-PreTraining): training from scratch with causal attention or recurrence.
- [MMLA-Kernels](https://github.com/MMLA-org/MMLA-Kernels): reference operators and experimental acceleration.
- MMLA architecture: [2606.28876v4](https://arxiv.org/abs/2606.28876v4).
- LARC: [2609.40063](https://arxiv.org/abs/2609.40063).
- MMLA-RSI: [2610.20294v1](https://arxiv.org/abs/2610.20294v1).

MMLA is mixer-agnostic. The supplied Transformer adapter is one projection of
its state interfaces. This library provides the recursive-rule runtime; it is
not a replacement for every architecture component or a complete pretrained model.

## Development and license

CPU checks cover rules, gradients, memory isolation, admission, interruptions,
recovery and version routing. CUDA performance and multi-GPU execution are not
qualified by those checks. Resource meters are cooperative; task backends need
host-enforced timeouts and process isolation for hard limits.

See [CONTRIBUTING.md](CONTRIBUTING.md). Code is licensed under **AGPL-3.0**;
model and dataset licenses remain separate. See [LICENSE](LICENSE).

### Project only the requested output positions

The default `AdapterProfile` continues to call the supplied causal LM's full
logits interface. For a decoder followed directly by a plain linear vocabulary
head, opt in with `SelectedHeadProfile(hidden_size=..., decoder_path="model",
head_path="lm_head")`. The decoder returns its fully normalized
`last_hidden_state`; the adapter selects requested positions before applying
the head. Full-prefix attention, masks, positions and input-LARC gradients stay
active. No KV or evicted-memory cache is introduced.

This explicit ABI is included in the profile and reader identity. Do not use it
for a model that rescales or postprocesses logits outside the head; use the
default adapter or implement that model's projection explicitly. CPU tests
check logits, gradients, sampling probabilities and reader invalidation.
GPU and end-to-end speed are workload dependent and are not claimed by these tests.
